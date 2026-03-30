#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

static std::string bbName(const BasicBlock &BB) {
  if (BB.hasName()) return std::string(BB.getName());
  return "<unnamed>";
}

static unsigned countMemoryOps(const BasicBlock &BB) {
  unsigned Cnt = 0;
  for (const Instruction &I : BB) {
    if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<AtomicRMWInst>(I) ||
        isa<AtomicCmpXchgInst>(I) || isa<AllocaInst>(I)) {
      ++Cnt;
    }
  }
  return Cnt;
}

struct DumpEdgeFeaturesPass : public PassInfoMixin<DumpEdgeFeaturesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    auto &BPI = FAM.getResult<BranchProbabilityAnalysis>(F);
    auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
    auto &LI  = FAM.getResult<LoopAnalysis>(F);

    errs() << "func,src_bb,dst_bb,succ_idx,prob_num,prob_den,"
           << "src_prof_count,dst_prof_count,edge_count_est,"
           << "src_inst_count,src_mem_ops,src_loop_depth,"
           << "is_back_edge,dst_loop_depth,dst_is_loop_header\n";

    for (BasicBlock &BB : F) {
      auto *TI = BB.getTerminator();
      if (!TI) continue;
      if (TI->getNumSuccessors() == 0) continue;

      uint64_t SrcCount = BFI.getBlockProfileCount(&BB).value_or(0);
      unsigned SrcInsts = BB.size();
      unsigned SrcMemOps = countMemoryOps(BB);
      unsigned SrcLoopDepth = LI.getLoopDepth(&BB);

      Loop *SrcLoop = LI.getLoopFor(&BB);

      for (unsigned i = 0; i < TI->getNumSuccessors(); ++i) {
        BasicBlock *Succ = TI->getSuccessor(i);
        if (!Succ) continue;

        BranchProbability P = BPI.getEdgeProbability(&BB, i);
        uint64_t DstCount = BFI.getBlockProfileCount(Succ).value_or(0);
        unsigned DstLoopDepth = LI.getLoopDepth(Succ);
        bool DstIsLoopHeader = LI.isLoopHeader(Succ);
        bool IsBackEdge = SrcLoop && Succ == SrcLoop->getHeader();

        uint64_t EdgeCountEst = 0;
        if (P.getDenominator() != 0) {
          EdgeCountEst =
              (uint64_t)((__uint128_t)SrcCount * P.getNumerator() / P.getDenominator());
        }

        errs() << F.getName() << ","
               << bbName(BB) << ","
               << bbName(*Succ) << ","
               << i << ","
               << P.getNumerator() << ","
               << P.getDenominator() << ","
               << SrcCount << ","
               << DstCount << ","
               << EdgeCountEst << ","
               << SrcInsts << ","
               << SrcMemOps << ","
               << SrcLoopDepth << ","
               << (IsBackEdge ? 1 : 0) << ","
               << DstLoopDepth << ","
               << (DstIsLoopHeader ? 1 : 0) << "\n";
      }
    }

    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION,
      "DumpEdgeFeaturesPass",
      "0.1",
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &FPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "dump-edge-features") {
                FPM.addPass(DumpEdgeFeaturesPass());
                return true;
              }
              return false;
            });
      }};
}
