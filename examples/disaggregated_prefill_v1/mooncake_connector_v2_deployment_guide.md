# Mooncake Connector V2 deployment guide

Mooncake Connector V2 discovers the local parallel topology from vLLM's
`parallel_config` and obtains the peer topology during the P/D metadata
handshake. Do not repeat P/D `tp_size`, `dp_size`, or `pp_size` values in
`kv_connector_extra_config`.

The vLLM service still needs its normal parallel command-line arguments. V2
removes only the duplicated connector-side topology configuration; it does not
choose the service parallel strategy.

## A5 GLM-Next example

The following example uses the field topology:

- Prefill: DP2 TP4 PP1
- Decode: DP4 TP2 PP1

Set the model path, network interfaces, device visibility, API addresses, and
DP RPC ports for the deployment environment.

### Prefill

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=<interface>
export TP_SOCKET_IFNAME=<interface>
export HCCL_SOCKET_IFNAME=<interface>

vllm serve <glm-next-model-path> \
  --host <prefill-host> \
  --port 8100 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --data-parallel-address <prefill-host> \
  --data-parallel-rpc-port 9100 \
  --trust-remote-code \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_buffer_device": "npu",
    "kv_role": "kv_producer",
    "engine_id": "glm-next-prefill",
    "kv_port": 20001
  }'
```

### Decode

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=<interface>
export TP_SOCKET_IFNAME=<interface>
export HCCL_SOCKET_IFNAME=<interface>

vllm serve <glm-next-model-path> \
  --host <decode-host> \
  --port 8200 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --data-parallel-address <decode-host> \
  --data-parallel-rpc-port 9200 \
  --trust-remote-code \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "engine_id": "glm-next-decode",
    "kv_port": 30001
  }'
```

For PP greater than one, add the normal `--pipeline-parallel-size` option to
the corresponding `vllm serve` command. It must not be copied into
`kv_connector_extra_config`.

`MooncakeConnectorV2` currently selects the pull implementation. The decode
worker fetches producer metadata using the request's `remote_host`,
`remote_port`, and `remote_engine_id`, then maps layers by complete layer name.

## Port range

For a service whose base port is `kv_port`, worker handshake ports occupy a
range derived from its real DP/PP/PCP/TP topology. Scheduler control ports are
allocated immediately after that range. Reserve enough consecutive ports for
the complete local topology and make them reachable between P and D nodes.

## Expected startup logs

At registration, V2 prints one summary for each distinct physical transfer
layout. For GLM-Next, verify that the main MLA and indexer-K entries retain the
same logical group but have different spec IDs, and that indexer-K reports:

```text
role=indexer_k, transfer_unit_tokens=128, block_lens=(8192,)
```

The expected A5 scales are 9 on Prefill and 17 on Decode. The indexer-state
entry must retain its independent logical group and report
`role=indexer_state`.
