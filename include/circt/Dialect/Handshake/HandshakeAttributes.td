//===- HandshakeAttributes.td - Attributes for Handshake ---*- tablegen -*-===//
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
    Reprensents a list of memory dependencies between the operation this
    attribute is attached to and other named memory accesses in the same
    function.
  }];

  let parameters = (ins ArrayRefParameter<"::circt::handshake::MemDependenceAttr">:$dependencies);
  let assemblyFormat = "`[` $dependencies `]`";
}

#endif // CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_TD
