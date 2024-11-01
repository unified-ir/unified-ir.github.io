---
sidebar_position: 1
---

# Unified IR

## About

**Uinfied IR** 是一套面向人工智能应用的统一中间表示（Unified Intermediate Representation），基于此可以方便的在不同的硬件平台上对人工智能应用做统一的编译优化和代码生成。随着深度学习在我们的日常生活中得到越来越广泛的应用，除英伟达外的各大硬件厂商也推出了自己的硬件加速平台用于大模型的训练与推理，比如：寒武纪、华为和沐曦。由于不同硬件平台的架构和编程接口不同，我们需要分别为它们编写庞大的深度学习算子库，这种依赖专家知识的方法通常不能得到所有算子的最佳实现。而且，各类对人工智能应用的编译工作往往和平台紧密耦合，难以在不同平台间进行迁移。因此，我们迫切需要一套能够对人工智能应用进行跨平台编译优化的统一方法。

**Unified IR** 正是为了解决这一困境而提出的，它是一个指令级的描述张量程序的中间表示，其设计的基本单元是对硬件指令的细粒度抽象，尽可能利用了各类硬件的架构和编程接口的共性，从而可以用一个统一的表示来描述各类硬件上的张量程序实现。通过在这样一个统一表示上用诸如 **IntelliGen** 这样的深度学习编译器对算子实现进行编译优化，然后将优化后的中间表示经过不同的后端生成对应硬件平台上的代码，我们可以很方便的为不同的硬件平台生成性能优异的算子实现。同时，这也意味着在 **Unified IR** 上所做的任何优化都可以无缝的迁移到所有支持的硬件平台上。

**Unified IR** 的优势有：
+ 用一种统一的方式来描述算子在不同硬件平台上的实现，从而可以实现对张量程序的跨平台编译优化；
+ 面向人工智能应用设计，相比于 OpenCL 这类具有强大表达能力的语言，更加充分的利用了深度学习算子的规整性，因此能挖掘并利用更多的优化机会，在表达能力和易优化性上取得了很好的平衡；
+ 在语义上贴近硬件的底层指令，可以十分方便的将其编译为不同硬件平台上的代码，具有很强的拓展性。

## Supported Platforms

目前，**Unified IR** 已经支持了NVIDIA、寒武纪和华为昇腾平台，我们通过深度学习编译器 **IntelliGen** 对张量程序的中间表示进行统一的优化，然后为不同的硬件平台实现了从 **Unified IR** 到对应平台代码生成后端，在包括Llama、GPT2的主流大模型中都取得了明显的加速。这样的结果表明，**Unified IR** 不仅可以为上层的人工智能应用提供恰当的抽象，从而能挖掘出更多的优化机会，还可以充分描述底层不同硬件平台的特性，能获得跨平台的优异性能。

### Results On Ascend 910B

![Ascend 910B上的加速效果，batch size = 1](/fig/overview/end-to-end-time-b1.png)

(batch size = 1)

![Ascend 910B上的加速效果，batch size = 4](/fig/overview/end-to-end-time.png)

(batch size = 4)
## Getting Started

在这里查看有关 **Unified IR** 的定义：[Introduction To Unified IR](https://github.com/deathwings602/Unified-IR/blob/main/doc/1-unified-ir-defs.md)

相关论文：[PowerFusion: A Tensor Compiler with Explicit Data Movement Description and Instruction-level Graph IR
](https://arxiv.org/abs/2307.04995)

