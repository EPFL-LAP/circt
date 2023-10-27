//===- HandshakeAttributes.td - Attributes for Handshake ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes modifications made as part of the Dynamatic project.
// See https://github.com/EPFL-LAP/dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file defines Handshake dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_TD
#define CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_TD

include "circt/Dialect/Handshake/Handshake.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

class Handshake_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Handshake_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Define an enumeration (along with a corresponding attribute) made up of two
// possible cases to represent the type (load or store) of a memory access.  
def AccessCaseLoad: I32EnumAttrCase<"Load", 0>;
def AccessCaseStore: I32EnumAttrCase<"Store", 1>;
def AccessTypeEnum: I32EnumAttr<
  "AccessTypeEnum", "Models a load or a store operation", [AccessCaseLoad, AccessCaseStore]> {
  let genSpecializedAttr = 0;
}
def AccessTypeEnumAttr: EnumAttr<Handshake_Dialect, AccessTypeEnum, "AccessType">;

// Represents memory accesses (in program order) to a memory region made from
// within a single block.
def BlockAccessesAttr : TypedArrayAttrBase<AccessTypeEnumAttr, "Array of memory operations">;

// Represents memory accesses (in program order) to a memory region made from
// multiple blocks (i.e., a function).
def FuncAccessesAttr : TypedArrayAttrBase<BlockAccessesAttr, "Array of array of memory operation">;

def MemAccessNameAttr : Handshake_Attr<"MemAccessName", "access_name"> {
  let summary = "A unique name given to a memory access";
  let description = [{
    This holds a unique name (within the context of an enclosing function) given
    to a memory operation. This is useful as part of memory dependency analysis
    to allow memory accesses to reference one another by name. This attribute is
    a simple wrapper around a StringAttr.
  }];
  
  let parameters = (ins "::mlir::StringAttr":$name);
  let builders = [
    AttrBuilder<(ins "::mlir::StringRef":$name),
      "return $_get(context, ::mlir::StringAttr::get(context, name));">,
  ];
  let assemblyFormat = "`<` $name `>`";
}

def DependenceComponentAttr : Handshake_Attr<"DependenceComponent", "dep_comp"> {
  let summary = "Information about the direction of a dependence for an AffineForOp";
  let description = [{
    Holds the same information as a `mlir::affine::DependenceComponent` (except 
    the `op` it refers to, since this would typically be attached to the
    operation in question or to one that would reference it through a
    `::circt::handshake::MemDependenceAttr`).
  }];

  let parameters = (ins "::std::optional<int64_t>":$lb, "::std::optional<int64_t>":$ub);
  let assemblyFormat = "`[` $lb `,` $ub `]`";
}

def MemDependenceAttr : Handshake_Attr<"MemDependence", "dep"> {
  let summary = "A memory dependence with respect to another memory access";
  let description = [{
    Represents a memory dependence between a source operation (the one this
    attribute is attached to) and a destination operation referenced through
    the `dstAccess` parameter. For example, if the source operation is a
    memref::StoreOp and the destination operation is a memref::LoadOp, this
    represents a RAW memory dependency. 
    
    The dependency is furthermore characterized by the loop depth at which the
    dependency is (`loopDepth`) and a list of dependence components
    (`components`) whose size indicates the number of commom loops surrounding
    both operations.
  }];
  
  let parameters = (ins 
    "::mlir::StringAttr":$dstAccess,
    "unsigned":$loopDepth,
    ArrayRefParameter<"::circt::handshake::DependenceComponentAttr">:$components
  );

  let builders = [
    AttrBuilder<(ins  "::mlir::StringRef":$dstAccess, 
                      "unsigned":$loopDepth,
                      "::mlir::ArrayRef<::mlir::affine::DependenceComponent>":$components), [{
      SmallVector<::circt::handshake::DependenceComponentAttr> compAttrs;
      for (auto &comp : components)
        compAttrs.push_back(::circt::handshake::DependenceComponentAttr::get(
          context, comp.lb, comp.ub));
      return $_get(context, ::mlir::StringAttr::get(context, dstAccess), 
        loopDepth, compAttrs);  
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    size_t getNumCommonLoops() {
      return getComponents().size();
    }
  }];
}

def MemDependenceArrayAttr : Handshake_Attr<"MemDependenceArray", "deps"> {
  let summary = "A list of memory dependencies with respect to other memory operations";
  let description = [{
    Represents a list of memory dependencies between the operation this
    attribute is attached to and other named memory accesses in the same
    function.
  }];

  let parameters = (ins ArrayRefParameter<"::circt::handshake::MemDependenceAttr">:$dependencies);
  let assemblyFormat = "`[` $dependencies `]`";
}

//===----------------------------------------------------------------------===//
// Channel annotations for buffer placement
//===----------------------------------------------------------------------===//

def ChannelBufPropsAttr : Handshake_Attr<"ChannelBufProps", "channelBufProps"> {
  let summary = "Buffering properties for a single channel";
  let description = [{
    Holds a set of buffering properties for a channel (i.e, an SSA value used
    only once) that are meant to be read by the buffer placement pass and
    constrain it in what kind of buffers it can place on the channel these
    properties are attached to. These properties are (1) the minimum and maximum
    number of transparent buffer slots placable on the channel, (2) the minimum
    and maximum number of opaque buffer slots placable on the channel, and (3)
    the channel's combinational delay expressed as a pair of pre-buffer
    (`inDelay`) and post-buffer (`outDelay`) delays as well as the total channel
    delay (`delay`) when it is unbuffered, all in ns.

    Users of this attribute may wish to interact with it through the
    dynamatic::ChannelBufProps struct, which represents the same information and
    is easier to manipulate than the raw MLIR attribute.
  }];
  
  let parameters = (ins "unsigned":$minTrans,
                        "std::optional<unsigned>":$maxTrans,
                        "unsigned":$minOpaque,
                        "std::optional<unsigned>":$maxOpaque,
                        "mlir::FloatAttr":$inDelay,
                        "mlir::FloatAttr":$outDelay,
                        "mlir::FloatAttr":$delay
                        );

  let builders = [
    AttrBuilder<(ins "dynamatic::ChannelBufProps&":$props), [{
      return $_get(context, props.minTrans, props.maxTrans, props.minOpaque, 
        props.maxOpaque, 
        FloatAttr::get(mlir::Float32Type::get(context), props.inDelay), 
        FloatAttr::get(mlir::Float32Type::get(context), props.outDelay),
        FloatAttr::get(mlir::Float32Type::get(context), props.delay));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Gets a ChannelBufProps instance that holds the same information as this
    /// attribute for easier data manipulation.
    inline dynamatic::ChannelBufProps getProps() {
      return dynamatic::ChannelBufProps(getMinTrans(), getMaxTrans(),
        getMinOpaque(), getMaxOpaque(), getInDelay().getValueAsDouble(),
        getOutDelay().getValueAsDouble(), getDelay().getValueAsDouble());
    }
  }];

  let hasCustomAssemblyFormat = 1;  
}

def OpBufPropsAttr : Handshake_Attr<"OpBufProps", "opBufProps"> {
  let summary = "Buffering properties for all channels defined by an operation";
  let description = [{
    Holds information about all specific buffering properties attached to the
    channels defined by an operation. In most situations, users of this
    attribute should use the helper functions in
    dynamatic/Transforms/BufferPlacement/BufferingStrategy.h to manipulate it,
    as its internal data structure is less convenient to use.

    The attribute tracks which buffering properties belong to which channel
    using each channel's (stringified) index in the definition list. Internally,
    it uses a map from stringified index to buffering properties.
  }];
  
  let parameters = (ins "mlir::DictionaryAttr":$channelProperties);

  let builders = [
    AttrBuilder<(ins 
      "mlir::SmallVector<NamedAttribute>&":$channelProperties), [{
      return $_get(context, 
        mlir::DictionaryAttr::get(context, channelProperties));   
    }]>
  ];

  let hasCustomAssemblyFormat = 1;  
  let genVerifyDecl = 1;
}

#endif // CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_TD
