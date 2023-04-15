//===- HandshakeAttributes.h - Declare Handshake attributes ------*- C++-*-===//
//
// This file defines Handshake dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_H
#define CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Handshake/HandshakeAttributes.h.inc"

#endif // CIRCT_DIALECT_HANDSHAKE_ATTRIBUTES_H
