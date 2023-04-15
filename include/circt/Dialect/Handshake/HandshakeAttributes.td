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

def AccessNameAttr : Handshake_Attr<"AccessName", "accessName"> {
  let summary = "A unique name given to a memory access";
  let description = [{
    TODO
  }];
  
  let parameters = (ins "::mlir::StringAttr":$name);
  let builders = [
    AttrBuilder<(ins "::mlir::StringRef":$name),
      "return $_get(context, ::mlir::StringAttr::get(context, name));">,
  ];
  let assemblyFormat = "`<` $name `>`";
}

def MemDependenceAttr : Handshake_Attr<"Dependency", "dep"> {
  let summary = "A memory dependency with respect to another named memory operation";
  let description = [{
    TODO
  }];
  
  let parameters = (ins "::mlir::StringAttr":$dst,
                        "::mlir::IntegerAttr":$numCommonLoops,
                        "::mlir::IntegerAttr":$loopDepth);

  let builders = [
    AttrBuilder<(ins "::mlir::StringRef":$name, "unsigned":$numCommonLoops, "unsigned":$loopDepth), [{
      auto signedness = ::mlir::IntegerType::SignednessSemantics::Unsigned;
      auto intType = ::mlir::IntegerType::get(context, 32, signedness);
      return $_get(context, ::mlir::StringAttr::get(context, name),
        IntegerAttr::get(intType, numCommonLoops), 
        IntegerAttr::get(intType, loopDepth));
      }]
    >,
  ];

  let assemblyFormat = "`<` $dst `(` $numCommonLoops `,` $loopDepth `)` `>`";
}

def MemDependenceArrayAttr : Handshake_Attr<"DependencyArray", "depArray"> {
  let summary = "A list of memory depedencies with respect to other memory operations";
  let description = [{
    TODO
  }];

  let parameters = (ins ArrayRefParameter<"::circt::handshake::DependencyAttr">:$dependencies);
  let assemblyFormat = "`[` $dependencies `]`";
}

#endif // CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_TD
