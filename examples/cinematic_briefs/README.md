# 可重复测试场景

本目录只保存稳定的 Cinematic Brief（六维输入），不保存 Agent、Scene IR、Blender 或渲染产物。所有新产物继续写入已被 Git Ignore 的 `runs/`。

## 荒漠飞船（10 秒）

```bash
cinescaffold run \
  --config .cinescaffold.conf \
  --brief examples/cinematic_briefs/desert_ship_10s.json \
  --output-dir runs/retest/desert_ship \
  --render-profile preview
```

## 嵌套太阳系（10 秒）

```bash
cinescaffold run \
  --config .cinescaffold.conf \
  --brief examples/cinematic_briefs/solar_system_10s.json \
  --output-dir runs/retest/solar_system \
  --render-profile preview
```

## 道路接车（12 秒）

```bash
cinescaffold run \
  --config .cinescaffold.conf \
  --brief examples/cinematic_briefs/roadside_pickup_12s.json \
  --output-dir runs/retest/roadside_pickup \
  --render-profile preview
```

为保持三组结果可比较，应在同一轮测试中冻结 `.cinescaffold.conf` 的 Provider、模型、thinking 模式和推理强度。只有专门调查 Provider 延迟时才在命令末尾增加 `--full-power-diagnostic`。
