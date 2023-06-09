# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ips_protocol/locatorevents.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ips_protocol/locatorevents.proto',
  package='ips.proto',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n ips_protocol/locatorevents.proto\x12\tips.proto\"D\n\x0cLocatorEvent\x12\t\n\x01t\x18\x01 \x02(\x01\x12)\n\x04type\x18\x02 \x02(\x0e\x32\x1b.ips.proto.LocatorEventType*l\n\x10LocatorEventType\x12\x13\n\x0f\x42UILDING_LOADED\x10\x00\x12\x15\n\x11\x46LIPMAP_GENERATED\x10\x01\x12\x14\n\x10NEW_LOCATOR_LOOP\x10\x02\x12\x16\n\x12KALMAN_INITIALIZED\x10\x03')
)

_LOCATOREVENTTYPE = _descriptor.EnumDescriptor(
  name='LocatorEventType',
  full_name='ips.proto.LocatorEventType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BUILDING_LOADED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLIPMAP_GENERATED', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NEW_LOCATOR_LOOP', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KALMAN_INITIALIZED', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=117,
  serialized_end=225,
)
_sym_db.RegisterEnumDescriptor(_LOCATOREVENTTYPE)

LocatorEventType = enum_type_wrapper.EnumTypeWrapper(_LOCATOREVENTTYPE)
BUILDING_LOADED = 0
FLIPMAP_GENERATED = 1
NEW_LOCATOR_LOOP = 2
KALMAN_INITIALIZED = 3



_LOCATOREVENT = _descriptor.Descriptor(
  name='LocatorEvent',
  full_name='ips.proto.LocatorEvent',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='t', full_name='ips.proto.LocatorEvent.t', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='ips.proto.LocatorEvent.type', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=47,
  serialized_end=115,
)

_LOCATOREVENT.fields_by_name['type'].enum_type = _LOCATOREVENTTYPE
DESCRIPTOR.message_types_by_name['LocatorEvent'] = _LOCATOREVENT
DESCRIPTOR.enum_types_by_name['LocatorEventType'] = _LOCATOREVENTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

LocatorEvent = _reflection.GeneratedProtocolMessageType('LocatorEvent', (_message.Message,), dict(
  DESCRIPTOR = _LOCATOREVENT,
  __module__ = 'ips_protocol.locatorevents_pb2'
  # @@protoc_insertion_point(class_scope:ips.proto.LocatorEvent)
  ))
_sym_db.RegisterMessage(LocatorEvent)


# @@protoc_insertion_point(module_scope)
