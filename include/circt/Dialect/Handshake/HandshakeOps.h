//===- Ops.h - Handshake MLIR Operations ------------------------*- C++ -*-===//
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

/// Forward declaration needed by `dynamatic::LoadPort`.
class DynamaticLoadOp;
/// Forward declaration needed by `dynamatic::StorePort`.
class DynamaticStoreOp;

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

/// Abstract base class for all memory ports. The class hierearchy supports
/// LLVM-style RTTI (i.e., isa/cast/dyn_cast) with optional-value casting, see
/// example below.
///
/// ```cpp
/// LoadPort loadPort (...);
/// SmallVector<MemoryPort> allPorts;
/// allPorts.push_back(loadPort);
/// std::optional<LoadPort> castedLoadPort = dyn_cast<LoadPort>(allPorts[0]);
/// assert(castedLoadPort);
/// std::optional<StorePort> notAStorePort = dyn_cast<StorePort>(allPorts[0]);
/// assert(!notAStorePort);
/// ```
class MemoryPort {
public:
  /// Kinds of memory ports (used for LLVM-style RTTI).
  enum class Kind {
    /// Control port
    CONTROL,
    /// Load port (handshake::DynamaticLoadOp)
    LOAD,
    /// Store port (handshake::DynamaticStoreOp)
    STORE
  };

  /// The operation producing the memory input(s) the port refers to.
  mlir::Operation *portOp;

  /// A `MemoryPort` cannot be instantiated directly (abstract class).
  MemoryPort() = delete;

  /// Default copy constructor.
  MemoryPort(const MemoryPort &memPort) = default;

  /// Returns the memory port's kind.
  Kind getKind() const { return kind; }

  /// Virtual default destructor.
  virtual ~MemoryPort() = default;

protected:
  /// List of indices in the operands/results of the memory interface the port
  /// refers to. Their meaning is port-kind-dependent.
  mlir::SmallVector<unsigned, 2> indices;

  /// Constructs a memory port "member-by-member".
  MemoryPort(mlir::Operation *accessOp, mlir::ArrayRef<unsigned> indices,
             Kind kind);

private:
  /// Memory port's kind (used for LLVM-style RTTI).
  Kind kind;
};

/// Memory control port which may be associated with any operation type
/// (typically, a constant to indicate a number of stores in a block or a
/// control-only value from a control merge).
class ControlPort : public MemoryPort {
public:
  /// Constructs the control port from any operation whose single result ends up
  /// as the memory input indicated by the index.
  ControlPort(mlir::Operation *ctrlOp, unsigned ctrlInputIdx);

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  ControlPort(const MemoryPort &memPort);

  /// Returns the control operation the port is associated to.
  mlir::Operation *getCtrlOp() const { return portOp; };

  /// Returns the index of the control port in its memory interface's inputs.
  unsigned getCtrlInputIdx() const { return indices[0]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::CONTROL;
  }
};

/// Memory load port associated with a `circt::handshake::DynamaticLoadOp`.
class LoadPort : public MemoryPort {
public:
  /// Constructs the load port from a load operation, the index of the load's
  /// address output in the memory interface's inputs, and the index of the
  /// load's data input in the memory interface's results.
  LoadPort(circt::handshake::DynamaticLoadOp loadOp, unsigned addrInputIdx,
           unsigned dataResIdx);

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  LoadPort(const MemoryPort &memPort);

  /// Returns the load operation the port is associated to.
  inline circt::handshake::DynamaticLoadOp getLoadOp() const;

  /// Returns the index of the port's address input in its memory interface's
  /// inputs.
  unsigned getAddrInputIdx() const { return indices[0]; }

  /// Returns the index of the port's data result in its memory interface's
  /// results.
  unsigned getDataResultIdx() const { return indices[1]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::LOAD;
  }
};

/// Memory store port associated with a `circt::handshake::DynamaticStoreOp`.
class StorePort : public MemoryPort {
public:
  /// Constructs the store port from a store operation and the index of the
  /// store's address output in the memory interface's inputs (the memory
  /// interface's input corresponding store's data output is assumed to be the
  /// next one).
  StorePort(circt::handshake::DynamaticStoreOp storeOp, unsigned baseInputIdx);

  /// Copy-constructor from abstract memory port for LLVM-style RTTI.
  StorePort(const MemoryPort &memPort);

  /// Returns the store operation the port is associated to.
  inline circt::handshake::DynamaticStoreOp getStoreOp() const;

  /// Returns the index of the port's address input in its memory interface's
  /// inputs.
  unsigned getAddrInputIdx() const { return indices[0]; }

  /// Returns the index of the port's data input in its memory interface's
  /// inputs.
  unsigned getDataInputIdx() const { return indices[1]; }

  /// Used by LLVM-style RTTI to establish `isa` relationships.
  static inline bool classof(const MemoryPort *port) {
    return port->getKind() == Kind::STORE;
  }
};

/// Represents a list of memory ports originating from a single basic block (in
/// the Handshake sense i.e., given by the attribute, not the actual MLIR
/// block) for a specific memory interface. A block may have a single control
/// port and 0 or more memory access ports (loads and stores) stored in the
/// same order as the memory interface's inputs.
class BlockMemoryPorts {
public:
  /// ID of the basic block the ports are contained in.
  unsigned blockID;
  /// Optional control port for the block.
  std::optional<ControlPort> ctrlPort;
  /// List of load/store accesses to the memory interface, ordered the same as
  /// the latter's inputs.
  mlir::SmallVector<MemoryPort> accessPorts;

  /// Initializes a block's memory ports without a control port (and with no
  /// access ports).
  BlockMemoryPorts(unsigned blockID);

  /// Initializes a block's memory ports with a control port (and with no access
  /// ports).
  BlockMemoryPorts(unsigned blockID, ControlPort ctrlPort);

  /// Whether the block's has a control port.
  inline bool hasControl() const { return ctrlPort.has_value(); }

  /// Computes the number of inputs in the asociated memory interface that map
  /// to this block's ports.
  unsigned getNumInputs() const;

  /// Computes the number of results in the asociated memory interface that map
  /// to this block's ports.
  unsigned getNumResults() const;

  /// Determines whether the block contains any port of the provided kind.
  bool hasAnyPort(MemoryPort::Kind kind) const;

  /// Determines the number of ports of the provided kind  the block contains.
  unsigned getNumPorts(MemoryPort::Kind kind) const;
};

/// Represents all memory ports originating from a Handshake function for a
/// specific memory interface. Ports are grouped by the basic block (in the
/// Handshake sense i.e., given by the attribute, not the actual MLIR block)
/// from which they originate. Groups of block ports are stored in the same
/// order as the memory interface's inputs. There may be 0 or more such groups.
class FuncMemoryPorts {
public:
  /// Memory interface associated with these ports.
  mlir::Operation *memInterfaceOp;
  /// List of blocks which contain at least one input port to the memory
  /// interface, ordered the same as the latter's inputs.
  mlir::SmallVector<BlockMemoryPorts> blocks;
  /// Bitwidth of control signals.
  unsigned ctrlWidth = 0;
  /// Bitwidth of address signals.
  unsigned addrWidth = 0;
  /// Bitwidth of data signals.
  unsigned dataWidth = 0;

  /// Initializes a function's memory ports from the memoryInterface it
  /// corresponds to (and without any port).
  FuncMemoryPorts(mlir::Operation *memInterfaceOp)
      : memInterfaceOp(memInterfaceOp){};

  /// Returns the continuous subrange of the memory interface's inputs which a
  /// block (indicated by its index in the list) maps to.
  mlir::ValueRange getBlockInputs(unsigned blockIdx);

  /// Returns the continuous subrange of the memory interface's results which a
  /// block (indicated by its index in the list) maps to.
  mlir::ValueRange getBlockResults(unsigned blockIdx);

  /// Returns the number of blocks with at least one port to the memory
  /// interface.
  unsigned getNumConnectedBlock() { return blocks.size(); }

  /// Determines whether the function contains any port of the provided kind.
  bool hasAnyPort(MemoryPort::Kind kind) const;

  /// Determines the number of ports of the provided kind the function contains.
  unsigned getNumPorts(MemoryPort::Kind kind) const;
};

/// Specifies how a handshake channel (i.e. a SSA value used once) may be
/// buffered. Backing data-structure for the ChannelBufPropsAttr attribute.
struct ChannelBufProps {
  /// Minimum number of transparent slots allowed on the channel (inclusive).
  unsigned minTrans;
  /// Maximum number of transparent slots allowed on the channel (inclusive).
  std::optional<unsigned> maxTrans;
  /// Minimum number of opaque slots allowed on the channel (inclusive).
  unsigned minOpaque;
  /// Maximum number of opaque slots allowed on the channel (inclusive).
  std::optional<unsigned> maxOpaque;
  /// Combinational delay (in ns) from the output port to the buffer's input, if
  /// a buffer is placed on the channel.
  double inDelay;
  /// Combinational delay (in ns) from the buffer's output to the input port, if
  /// a buffer is placed on the channel.
  double outDelay;
  /// Total combinational channel delay (in ns) if no buffer is placed on the
  /// channel.
  double delay;

  /// Simple constructor that takes the same parameters as the struct's members.
  /// By default, all the channel is "unconstrained" w.r.t. what kind of buffers
  /// can be placed and is assumed to have 0 delay.
  ChannelBufProps(unsigned minTrans = 0,
                  std::optional<unsigned> maxTrans = std::nullopt,
                  unsigned minOpaque = 0,
                  std::optional<unsigned> maxOpaque = std::nullopt,
                  double inDelay = 0.0, double outDelay = 0.0,
                  double delay = 0.0);

  /// Determines whether these buffering properties are satisfiable i.e.,
  /// whether it's possible to create a buffer that respects them.
  bool isSatisfiable() const;

  /// Determines whether these buffering properties forbid the placement of any
  /// buffer on the associated channel.
  bool isBufferizable() const;

  /// Computes member-wise equality.
  bool operator==(const ChannelBufProps &rhs) const;
};
} // namespace dynamatic

static inline std::string getMaxStr(std::optional<unsigned> optMax) {
  return optMax.has_value() ? (std::to_string(optMax.value()) + "]") : "inf]";
};

namespace llvm {
/// Struct to enable LLVM-style RTTI for the class hierarchy under
/// `dynamatic::MemoryPort`.
template <typename T>
struct CastInfo<T, dynamatic::MemoryPort>
    : OptionalValueCast<T, dynamatic::MemoryPort> {};

} // namespace llvm

/// Prints the buffering properties as two closed or semi-open intervals
/// (depending on whether maximums are defined), one for tranparent slots and
/// one for opaque slots.
template <typename Os>
Os &operator<<(Os &os, dynamatic::ChannelBufProps &props) {
  os << "{\n\ttransparent slots: [" << props.minTrans << ", "
     << getMaxStr(props.maxTrans) << "\n\topaque slots: [" << props.minOpaque
     << ", " << getMaxStr(props.maxOpaque) << "\n\tin/out delays: ("
     << props.inDelay << ", " << props.outDelay << ")"
     << "\n\ttotal delay: " << props.delay << "\n}\n";
  return os;
}

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
