//===- Ops.h - Handshake MLIR Operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_HANDSHAKEOPS_OPS_H_
#define CIRCT_HANDSHAKEOPS_OPS_H_

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Any.h"

namespace circt {
namespace handshake {

struct MemLoadInterface {
  unsigned index;
  mlir::Value addressIn;
  mlir::Value dataOut;
  mlir::Value doneOut;
};

struct MemStoreInterface {
  unsigned index;
  mlir::Value addressIn;
  mlir::Value dataIn;
  mlir::Value doneOut;
};

/// Default implementation for checking whether an operation is a control
/// operation. This function cannot be defined within ControlInterface
/// because its implementation attempts to cast the operation to an
/// SOSTInterface, which may not be declared at the point where the default
/// trait's method is defined. Therefore, the default implementation of
/// ControlInterface's isControl method simply calls this function.
bool isControlOpImpl(Operation *op);

#include "circt/Dialect/Handshake/HandshakeInterfaces.h.inc"

} // end namespace handshake
} // end namespace circt

namespace dynamatic {

/// Specifies how a handshake channel (i.e. a SSA value used once) may be
/// buffered. Backing data-structure for the ChannelBufPropsAttr attribute.
struct ChannelBufProps {
  /// Minimum number of transparent slots allowed on the channel
  unsigned minTrans;
  /// Maximum number of transparent slots allowed on the channel
  std::optional<unsigned> maxTrans;
  /// Minimum number of opaque slots allowed on the channel
  unsigned minOpaque;
  /// Maximum number of opaque slots allowed on the channel
  std::optional<unsigned> maxOpaque;

  /// Simple constructor that takes the same parameters as the struct's members.
  /// By default, all parameters are set so that the channel is "unconstrained"
  /// w.r.t. what kind of buffers can be placed.
  ChannelBufProps(unsigned minTrans = 0,
                  std::optional<unsigned> maxTrans = std::nullopt,
                  unsigned minOpaque = 0,
                  std::optional<unsigned> maxOpaque = std::nullopt);

  /// Determines whether these buffering properties are satisfiable i.e.,
  /// whether it's possible to create a buffer that respects them.
  bool isSatisfiable() const;

  /// Computes member-wise equality.
  bool operator==(const ChannelBufProps &rhs) const;
};

/// Custom specialization of llvm::hash_value for ChannelBufProps. Converts the
/// struct to a tuple and use the hash_value function on tuples to get our own
/// hash.
// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const ChannelBufProps &props);
} // namespace dynamatic

/// Prints the buffering properties as two closed or semi-open intervals
/// (depending on whether maximum are defined), one for tranparent slots and one
/// for opaque slots.
std::ostream &operator<<(std::ostream &os,
                         const dynamatic::ChannelBufProps &props);

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
class HasClock : public TraitBase<ConcreteType, HasClock> {};
} // namespace OpTrait

namespace affine {
struct DependenceComponent;
} // namespace affine
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Handshake/HandshakeAttributes.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/Handshake.h.inc"

#endif // MLIR_HANDSHAKEOPS_OPS_H_
