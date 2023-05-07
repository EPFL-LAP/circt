//===- HandshakeDialect.cpp - Implement the Handshake dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::handshake;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void HandshakeDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Handshake/Handshake.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Handshake/HandshakeAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Dialect attributes method definitions
//===----------------------------------------------------------------------===//

// LSQBlockAttr

void LSQBlockAttr::print(AsmPrinter &odsPrinter) const {
  auto accesses = getAccesses();
  odsPrinter << "[";
  for (auto &acc : accesses.drop_back(1))
    odsPrinter << stringifyAccessTypeEnum(acc) << ", ";
  odsPrinter << stringifyAccessTypeEnum(accesses.back()) << "]";
}

Attribute LSQBlockAttr::parse(AsmParser &odsParser, Type odsType) {
  SmallVector<AccessTypeEnum> accesses;
  auto parseAccess = [&]() -> ParseResult {
    std::string accessTypeStr;
    if (odsParser.parseString(&accessTypeStr))
      return failure();

    auto accessType = symbolizeAccessTypeEnum(accessTypeStr);
    if (!accessType.has_value())
      return failure();

    accesses.push_back(accessType.value());
    return success();
  };

  if (odsParser.parseLSquare() ||
      odsParser.parseCommaSeparatedList(parseAccess) ||
      odsParser.parseRSquare())
    return Attribute();

  return odsParser.getChecked<LSQBlockAttr>(odsParser.getContext(), accesses);
}

LogicalResult LSQBlockAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   ArrayRef<AccessTypeEnum> accesses) {
  if (accesses.empty())
    return emitError() << "LSQ block must have at least one access";
  return success();
}

// LSQBlocksAttr

LogicalResult
LSQBlocksAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<LSQBlockAttr> blocks) {
  if (blocks.empty())
    return emitError() << "LSQ must be connected to at least one block";
  return success();
}

// MemDependenceAttr

/// Pretty-prints a dependence component as [lb, ub] where both lb and ub
/// are optional (in which case, no bound is printed).
static void printDependenceComponent(AsmPrinter &odsPrinter,
                                     const DependenceComponentAttr &comp) {
  std::string lb =
      comp.getLb().has_value() ? std::to_string(comp.getLb().value()) : "";
  std::string ub =
      comp.getUb().has_value() ? std::to_string(comp.getUb().value()) : "";
  odsPrinter << "[" << lb << ", " << ub << "]";
}

/// Parses one optional bound from a dependence component. Succeeds when parsing
/// succeeds (even if the value is not present, since it's optional).
static ParseResult parseDependenceComponent(AsmParser &odsParser,
                                            std::optional<int64_t> &opt) {
  // For some obscure reason, trying to directly parse an int64_t from the IR
  // results in errors when the bound is either INT64_MIN or INT64_MAX. We
  // hack around this by parsing APInt instead and then explicitly checking for
  // underflow/overflow before converting back to an int64_t
  APInt bound;
  if (auto res = odsParser.parseOptionalInteger(bound); res.has_value()) {
    if (failed(res.value()))
      return failure();
    if (bound.getBitWidth() > sizeof(int64_t) * CHAR_BIT)
      opt = bound.isNegative() ? INT64_MIN : INT64_MAX;
    else
      opt = bound.getSExtValue();
  }
  return success();
}

void MemDependenceAttr::print(AsmPrinter &odsPrinter) const {
  // Print destination memory access and loop depth
  odsPrinter << "<\"" << getDstAccess().str() << "\" (" << getLoopDepth()
             << ")";

  // Print dependence components, if present
  auto components = getComponents();
  if (!components.empty()) {
    odsPrinter << " [";
    for (auto &comp : components.drop_back(1)) {
      printDependenceComponent(odsPrinter, comp);
      odsPrinter << ", ";
    }
    printDependenceComponent(odsPrinter, components.back());
    odsPrinter << "]";
  }
  odsPrinter << ">";
}

Attribute MemDependenceAttr::parse(AsmParser &odsParser, Type odsType) {

  MLIRContext *ctx = odsParser.getContext();

  // Parse destination memory access
  std::string dstName;
  if (odsParser.parseLess() || odsParser.parseString(&dstName))
    return Attribute();
  auto dstAccess = StringAttr::get(ctx, dstName);

  // Parse loop depth
  unsigned loopDepth;
  if (odsParser.parseLParen() || odsParser.parseInteger<unsigned>(loopDepth) ||
      odsParser.parseRParen())
    return Attribute();

  // Parse dependence components if present
  SmallVector<DependenceComponentAttr> components;
  if (!odsParser.parseOptionalLSquare()) {
    auto parseDepComp = [&]() -> ParseResult {
      std::optional<int64_t> optLb = std::nullopt, optUb = std::nullopt;

      // Parse [lb, ub] where both lb and ub are optional
      if (odsParser.parseLSquare() ||
          failed(parseDependenceComponent(odsParser, optLb)) ||
          odsParser.parseComma() ||
          failed(parseDependenceComponent(odsParser, optUb)) ||
          odsParser.parseRSquare())
        return failure();

      components.push_back(DependenceComponentAttr::get(ctx, optLb, optUb));
      return success();
    };

    if (odsParser.parseCommaSeparatedList(parseDepComp) ||
        odsParser.parseRSquare())
      return Attribute();
  }

  if (odsParser.parseGreater())
    return Attribute();
  return MemDependenceAttr::get(ctx, dstAccess, loopDepth, components);
}

// Provide implementations for the enums, attributes and interfaces that we use.
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Handshake/HandshakeAttributes.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeDialect.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeEnums.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
