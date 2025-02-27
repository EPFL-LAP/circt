# REQUIRES: esi-runtime, esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim.py -- %PYTHON% %S/test_software/esi_test.py cosim env

import pycde
from pycde import (AppID, Clock, Input, Module, generator)
from pycde.bsp import cosim
from pycde.constructs import Wire
from pycde.esi import ServiceDecl
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, UInt)

import sys

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.TO, UInt(16)),
    BundledChannel("req", ChannelDirection.FROM, UInt(24))
])


@ServiceDecl
class HostComms:
  req_resp = TestBundle


class LoopbackInOutAdd7(Module):
  """Loopback the request from the host, adding 7 to the first 15 bits."""

  @generator
  def construct(ports):
    loopback = Wire(Channel(UInt(16)))
    call_bundle, [(_, from_host)] = TestBundle.pack(resp=loopback)
    HostComms.req_resp(call_bundle, AppID("loopback_inout"))

    ready = Wire(Bits(1))
    data, valid = from_host.unwrap(ready)
    plus7 = data + 7
    data_chan, data_ready = loopback.type.wrap(plus7.as_uint(16), valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class Top(Module):
  clk = Clock()
  rst = Input(Bits(1))

  @generator
  def construct(ports):
    LoopbackInOutAdd7()


if __name__ == "__main__":
  s = pycde.System(cosim.CosimBSP(Top),
                   name="ESILoopback",
                   output_directory=sys.argv[1])
  s.compile()
  s.package()

  s.print()
