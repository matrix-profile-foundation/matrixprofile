To update the messages module, you must run:

```
protoc proto_messages.proto --python_out=.
```

This regenerates the "proto_messages_pb2.py" module. It should **NEVER** be edited directly. Also note that this requires you to have the protobuf compiler installed on your system.

https://github.com/protocolbuffers/protobuf