#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

typedef std::pair<BasicBlock*, BasicBlock*> edge;

typedef struct BBFeatures {
  long numInstr = 0;           // Done
  long loopDepth = 0;          // Done
  long finalOpcode = -1;       // Done
  long numSuccessors = 0;      // Done
  double fracLoads = 0;        // Done
  double fracStores = 0;       // Done
  double fracAllocs = 0;       // Done
  double fracOtherMem = 0;     // Done
  double fracArith = 0;        // Done
  double fracIArith = 0;       // Done
  double fracFArith = 0;       // Done
  double fracCalls = 0;        // Done
  double fracCasts = 0;        // Done
  long numICmps = 0;           // Done
  long numPtrCmps = 0;         // Done
  long numBothPtrUnknown = 0;  // Done
  long numPtrCmpNull = 0;      // Done
  long numPtrCmpOtherConst = 0;
  long numFCmps = 0;               // Done
  long numCmpsZero = 0;            // Done
  long numCmpsNegOne = 0;          // Done
  long numCmpsOtherConst = 0;      // Done
  long numCmpsBothUnknown = 0;     // Done
  double fracPhiNodes = 0;         // Done
  long maxPhiIncoming = 0;         // Done
  double avgPhiIncomingEdges = 0;  // Done
  long numParentBlocks = 0;        // Done
  double avgValueUsage = 0;
  long maxValueUsage = 0;      // Done
  long numVarsUsed = 0;        // Done
  long entranceDistance = -1;  // Done
  long exitDistance = -1;
  // TODO: distance from entrance/exit,
} BBFeatures;

typedef std::unordered_map<BasicBlock*, BBFeatures> FeatureMap;

typedef struct EdgeFeatures {
  bool isBackEdge;
  bool isBranch;
  long loopDepthJumped;
} EdgeFeatures;

typedef enum ICmpType {
  PtrNull,
  PtrConst,
  PtrPtr,
  IntZero,
  IntNegOne,
  IntConst,
  IntInt,
} ICmpType;

typedef enum FCmpType { FloatZero, FloatConst, FloatFloat } FCmpType;

typedef std::unordered_map<BasicBlock*, long> distanceMap;

distanceMap getEntranceDistances(Function& F) {
  distanceMap distances;
  std::queue<std::pair<BasicBlock*, long>> bfsQueue;
  bfsQueue.push({&F.getEntryBlock(), 0});
  while (!bfsQueue.empty()) {
    auto curr = bfsQueue.front();
    distances[curr.first] = curr.second;
    long nextDistance = curr.second + 1;
    for (auto succ : successors(curr.first)) {
      bfsQueue.push({succ, nextDistance});
    }
    bfsQueue.pop();
  }
  return distances;
}

distanceMap getExitDistances(Function& F) {
  distanceMap distances;
  std::queue<std::pair<BasicBlock*, long>> bfsQueue;
  // Find blocks that exit the function based on terminating instruction
  for (auto& BB : F) {
    if (auto term = BB.getTerminator()) {
      if (term->willReturn()) {
        bfsQueue.push({&BB, 0});
      }
    }
  }
  while (!bfsQueue.empty()) {
    auto curr = bfsQueue.front();
    distances[curr.first] = curr.second;
    long nextDistance = curr.second + 1;
    for (auto pred : predecessors(curr.first)) {
      bfsQueue.push({pred, nextDistance});
    }
    bfsQueue.pop();
  }
  return distances;
}

ICmpType getICmpType(const Instruction& I) {
  auto left = I.getOperand(0);
  auto right = I.getOperand(1);
  // TODO: is this the correct way to check for pointerness?
  bool isPtrCmp = isa<PointerType>(left->getType());
  llvm::Constant* constant = dyn_cast<llvm::Constant>(left);
  if (constant == NULL) {
    constant = dyn_cast<llvm::Constant>(right);
  }
  if (isPtrCmp) {
    if (constant) {
      return constant->isNullValue() ? ICmpType::PtrNull : ICmpType::PtrConst;
    }
    return ICmpType::PtrPtr;
  } else {
    if (constant) {
      if (constant->isZeroValue()) {
        return ICmpType::IntZero;
      } else if (constant->getUniqueInteger() == -1) {
        return ICmpType::IntNegOne;
      } else {
        return ICmpType::IntConst;
      }
    }
    return ICmpType::IntInt;
  }
}

FCmpType getFCmpType(const Instruction& I) {
  auto left = I.getOperand(0);
  auto right = I.getOperand(1);
  llvm::Constant* constant = dyn_cast<llvm::Constant>(left);
  if (constant) {
    if (constant->isZeroValue()) {
      return FCmpType::FloatZero;
    }
    return FCmpType::FloatConst;
  }
  return FCmpType::FloatFloat;
}

static std::string bbName(const BasicBlock& BB) {
  if (BB.hasName()) return std::string(BB.getName());
  return "<unnamed>";
}

static BBFeatures getBlockFeatures(const BasicBlock& BB) {
  BBFeatures feats;
  // CFG Features Accessible from Block
  feats.numSuccessors = 0;
  for (auto _ : successors(&BB)) {
    feats.numSuccessors++;
  }
  feats.numParentBlocks = 0;
  for (auto _ : predecessors(&BB)) {
    feats.numParentBlocks++;
  }
  if (auto lastInstruction = BB.getTerminator()) {
    feats.finalOpcode = lastInstruction->getOpcode();
  } else {
    feats.finalOpcode = -1;
  }

  std::unordered_set<Value*> usedValues;
  // Instruction Level feature setup
  feats.numInstr = BB.size();
  long load = 0, store = 0, otherMem = 0, alloc = 0, arith = 0, floatArith = 0,
       intArith = 0, logic = 0, call = 0, cast = 0, cmpPtrPtr = 0, phiNodes = 0,
       totalPhiIncoming = 0, totalValueUses = 0;
  // Initialize things we update within the struct as we iterate
  feats.numICmps = 0;
  feats.numPtrCmps = 0;
  feats.numBothPtrUnknown = 0;
  feats.numFCmps = 0;
  feats.numCmpsOtherConst = 0;
  feats.numCmpsBothUnknown = 0;
  feats.numCmpsZero = 0;
  feats.numPtrCmpNull = 0;
  feats.numCmpsNegOne = 0;
  feats.maxPhiIncoming = 0;
  feats.maxValueUsage = 0;
  // Big loop --- most of the features are collected here, but its a bit gross
  for (const Instruction& I : BB) {
    // Value usage for instruction across function
    totalValueUses += I.getNumUses();
    if (I.getNumUses() > feats.maxValueUsage) {
      feats.avgValueUsage = I.getNumUses();
    }
    for (const Use& op : I.operands()) {
      usedValues.insert(op.get());
    }
    // Count Operation Types
    if (isa<LoadInst>(I)) {
      load++;
    } else if (isa<StoreInst>(I)) {
      store++;
    } else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I)) {
      otherMem++;
    } else if (isa<AllocaInst>(I)) {
      alloc++;
    } else if (I.getOpcode() <= 24 && I.getOpcode() >= 13 ||
               I.isArithmeticShift()) {
      // Arithmetic ops are mostly all contiguous, except for the shifts.
      arith++;
      switch (I.getOpcode()) {
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Mul:
        case Instruction::UDiv:
        case Instruction::SDiv:
        case Instruction::AShr:
          intArith++;
          break;
        default:
          floatArith++;
          break;
      }
    } else if (I.isBitwiseLogicOp() || I.isLogicalShift()) {
      logic += 1;
    } else if (isa<CallInst>(I) || isa<CallBrInst>(I)) {
      call++;
    } else if (I.isCast()) {
      cast++;
    } else if (isa<CmpInst>(I)) {
      // There are many features to check here
      if (I.getOpcode() == Instruction::ICmp) {
        // ICmp also is used for comparing pointers
        switch (getICmpType(I)) {
          case PtrConst:
            feats.numPtrCmpOtherConst++;
            feats.numPtrCmps++;
            break;
          case PtrNull:
            feats.numPtrCmpNull++;
            feats.numPtrCmps++;
            break;
          case PtrPtr:
            cmpPtrPtr++;
            feats.numBothPtrUnknown++;
            feats.numPtrCmps++;
            break;
          case IntZero:
            feats.numCmpsZero++;
            feats.numICmps++;
            break;
          case IntNegOne:
            feats.numCmpsNegOne++;
            feats.numICmps++;
            break;
          case IntConst:
            feats.numCmpsOtherConst++;
            feats.numICmps++;
            break;
          case IntInt:
            feats.numCmpsBothUnknown++;
            feats.numICmps++;
            break;
        }
      } else if (I.getOpcode() == Instruction::FCmp) {
        feats.numFCmps++;
        switch (getFCmpType(I)) {
          case FloatZero:
            feats.numCmpsZero++;
            break;
          case FloatConst:
            feats.numCmpsOtherConst++;
            break;
          case FloatFloat:
            feats.numCmpsBothUnknown++;
            break;
        }
      } else if (isa<PHINode>(I)) {
        auto incoming = I.getNumOperands();
        if (incoming > feats.maxPhiIncoming) {
          feats.maxPhiIncoming = incoming;
        }
        totalPhiIncoming += incoming;
        phiNodes++;
      }
    }
  }
  // Calculate all the fractional things based on the counts
  double size = static_cast<double>(BB.size());
  feats.fracLoads = load / size;
  feats.fracStores = store / size;
  feats.fracAllocs = alloc / size;
  feats.fracOtherMem = otherMem / size;
  feats.fracArith = arith / size;
  feats.fracIArith = intArith / size;
  feats.fracFArith = floatArith / size;
  feats.fracPhiNodes = phiNodes / size;
  feats.fracCalls = call / size;
  feats.fracCasts = cast / size;
  feats.avgPhiIncomingEdges = totalPhiIncoming / static_cast<double>(phiNodes);
  feats.numVarsUsed = usedValues.size();

  return feats;
}

struct DumpEdgeFeaturesPass : public PassInfoMixin<DumpEdgeFeaturesPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager& FAM) {
    auto& BPI = FAM.getResult<BranchProbabilityAnalysis>(F);
    auto& BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
    auto& LI = FAM.getResult<LoopAnalysis>(F);
    distanceMap entranceDistances = getEntranceDistances(F);
    distanceMap exitDistances = getExitDistances(F);
    FeatureMap blockFeatureMap;

    // Old printing -- not yet updated!
    // errs() << "func,src_bb,dst_bb,succ_idx,prob_num,prob_den,"
    //        << "src_prof_count,dst_prof_count,edge_count_est,"
    //        << "src_inst_count,src_mem_ops,src_loop_depth,"
    //        << "is_back_edge,dst_loop_depth,dst_is_loop_header\n";

    for (BasicBlock& BB : F) {
      auto* TI = BB.getTerminator();
      if (!TI) continue;
      if (TI->getNumSuccessors() == 0) continue;

      BBFeatures feats = getBlockFeatures(BB);
      feats.loopDepth = LI.getLoopDepth(&BB);
      if (entranceDistances.count(&BB)) {
        feats.entranceDistance = entranceDistances.at(&BB);
      } else {
        feats.entranceDistance = -1;
      }
      if (exitDistances.count(&BB)) {
        feats.exitDistance = exitDistances.at(&BB);
      } else {
        feats.exitDistance = -1;
      }
      blockFeatureMap[&BB] = feats;

      uint64_t SrcCount = BFI.getBlockProfileCount(&BB).value_or(0);
      unsigned SrcInsts = BB.size();
      unsigned SrcLoopDepth = LI.getLoopDepth(&BB);

      Loop* SrcLoop = LI.getLoopFor(&BB);

      for (unsigned i = 0; i < TI->getNumSuccessors(); ++i) {
        BasicBlock* Succ = TI->getSuccessor(i);
        if (!Succ) continue;

        BranchProbability P = BPI.getEdgeProbability(&BB, i);
        uint64_t DstCount = BFI.getBlockProfileCount(Succ).value_or(0);
        unsigned DstLoopDepth = LI.getLoopDepth(Succ);
        bool DstIsLoopHeader = LI.isLoopHeader(Succ);
        bool IsBackEdge = SrcLoop && Succ == SrcLoop->getHeader();

        uint64_t EdgeCountEst = 0;
        if (P.getDenominator() != 0) {
          EdgeCountEst = (uint64_t)((__uint128_t)SrcCount * P.getNumerator() /
                                    P.getDenominator());
        }

        // Old printing --- not yet updated!
        // errs() << F.getName() << "," << bbName(BB) << "," << bbName(*Succ)
        //        << "," << i << "," << P.getNumerator() << ","
        //        << P.getDenominator() << "," << SrcCount << "," << DstCount
        //        << "," << EdgeCountEst << "," << SrcInsts << "," << SrcMemOps
        //        << "," << SrcLoopDepth << "," << (IsBackEdge ? 1 : 0) << ","
        //        << DstLoopDepth << "," << (DstIsLoopHeader ? 1 : 0) << "\n";
      }
    }

    return PreservedAnalyses::all();
  }
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "DumpEdgeFeaturesPass", "0.1",
          [](PassBuilder& PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager& FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "dump-edge-features") {
                    FPM.addPass(DumpEdgeFeaturesPass());
                    return true;
                  }
                  return false;
                });
          }};
}
