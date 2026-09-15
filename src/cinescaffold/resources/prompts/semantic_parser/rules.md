# Cinematic Brief 语义契约 v0.13

## 1. 术语与职责

以下术语在本契约中含义固定：

- **视频片段**：用户描述的一个连续输出及其完整时间范围。
- **场景实体**：需要独立识别、定位、运动、构图或参与关系的对象，对应 `subjects`。
- **动作阶段**：某个场景实体在一个数值时间区间内保持或改变状态，对应一条 `subject_motion`。
- **摄影机**：观察场景的虚拟摄影机；其位置、朝向和运动写入 `camera`。
- **光学属性**：焦距、景深等摄影机光学意图，写入 `camera.lens_intent`。
- **画面构图**：投影后的景别、画面位置、视觉占比和可见性要求，写入 `composition`。
- **空间关系**：两个场景实体之间的相对位置或尺度事实，写入 `scene_design.relationships`。

自然语言中的同一个词可能有多种意思，必须按上下文归入上述唯一维度。不得用含义不明的“主体”混指主要对象、施事者或所有对象；不得用含义不明的“镜头”混指视频片段、摄影机、光学属性或画面构图。JSON 字段名保持 Schema 规定，不因此改名。

Semantic 阶段负责忠实建模语义，不负责生成三维坐标、关键帧、摄影机矩阵、资产替换或渲染参数。数值缺省和量化由应用中的版本化规则处理，不在这里凭常识估算。

## 2. 原文证据与不确定性

所有带来源字段的值必须遵循：

- `explicit`：原文直接表达该值；`source_text` 保存支持它的最小原文片段。
- `inferred`：该值能由原文事实必然推出，而不是常见理解、较高概率或创作偏好；必须保留证据片段。
- `default`：本契约或 Schema 明确要求的结构缺省；`source_text` 为 `null`。
- `unknown`：原文不足以决定；值允许为空时应保持为空，`source_text` 为 `null`。

会影响几何、时间、方向、目标、路径、构图或摄影机行为的多种合理解释，不得擅自选择。将缺失点写入 `uncertainties`：未选择时用 `unresolved`；应用规则明确给出缺省时用 `use_default`；只有必然推断时用 `use_inference`。摘要不得把推断、缺省或未知内容改写成用户原话。

用户明确表达的要求优先。不得为了填满字段，复制无关原文作为证据；不得从实体名称、单个汉字、语言子串或故事常识推出方向、目标、载体或时间关系。

## 3. 场景实体与环境

- 为每个需要独立定位、运动、构图或参与关系的对象创建一个稳定且唯一的 `subjects[].id`。
- 一个地点名词若只表示环境，写入 `scene_design.environment`；若其中对象还需要独立定位、运动、构图或参与关系，也必须创建场景实体。
- 只保留原文给出的类别、描述、叙事作用和属性。不能确定的属性不补写。
- `spatial_layers` 表达原文明确的前景、中景、后景等画面纵深分层，并引用已有场景实体。
- `environmental_motion` 只描述环境本身的动态，不代替场景实体的动作阶段。

`scene_dynamics` 只判断场景实体的可观察状态是否改变：位置、姿态、尺度、可见性或容纳状态在视频片段内发生任何变化即为 `dynamic`；全部保持不变即为 `static`。摄影机运动、焦距变化和纯环境效果不改变该分类。分类必须与所有动作阶段的类型字段一致。

## 4. 动作阶段与类型字段

每个场景实体独立建立稀疏时间线。一个动作或状态具有独立起止范围时，必须建立独立的 `subject_motion`，并提供唯一 `motion_id`、有效 `subject_id` 和数值起止时间。动作阶段是语义区间，不是逐帧动画关键帧。

`action.value` 仅用于人类阅读和原文审计。以下字段共同构成下游唯一使用的机器语义，必须互相一致：

### 4.1 动作与驱动方式

- `motion_mode=stationary`：没有整体位移；必须搭配 `action_kind=hold`、`motion_type=static`、`path_type=stationary`。
- `motion_mode=local_interaction`：只有局部姿态或局部状态变化；必须搭配 `action_kind=interact`、`motion_type=interactive`、`path_type=stationary`。
- `motion_mode=self_propelled`：场景实体自主产生整体位移；必须搭配 `action_kind=locomotion`，`motion_type` 从 `walking/running/flying/jumping/moving` 中选择，路径不得为 `stationary`。
- `motion_mode=carried`：场景实体的世界运动由另一个场景实体承载；必须搭配 `action_kind=locomotion`、`motion_type=carried`、非空 `carrier_id`、`direction_mode=none`、`target_id=null`、`path_type=stationary`。该阶段不重复生成自主世界路径。

`action_kind=other` 是 Schema 保留值，当前输出不得使用；无法分类时必须在 `uncertainties` 中保留问题，而不是绕过类型字段的一致性。`carrier_id` 仅在 `carried` 时填写，且必须引用另一个已有场景实体。

### 4.2 方向、目标与路径

- `direction_mode=none`：原文没有可执行的几何方向；`target_id` 必须为 `null`。
- `direction_mode=world_forward`：原文明确要求沿世界前方；`target_id` 必须为 `null`。
- `direction_mode=toward_target`：明确朝某场景实体接近；必须填写该实体的 `target_id`。
- `direction_mode=away_from_target`：明确背离某场景实体；必须填写该实体的 `target_id`。
- `direction_mode=relative_to_target`：明确以某场景实体为相对参考执行路径；必须填写该实体的 `target_id`。

动作的参与者、受影响对象、叙事关注对象和几何运动目标不是同一概念。仅当原文明确提供几何关系时才能填写非 `none` 方向和目标。普通的出现、经过、到达或离开只证明状态或位置发生变化，不自动证明相对于哪个场景实体运动。

`path_type` 只在原文提供足够证据时选用 `linear/circular/elliptical/s_curve/figure_eight/parabolic`；自主位移但路径形状未定时用 `unspecified`。不得发明某个故事专用动作类型或关系类型。

`postconditions` 只描述该动作阶段结束后的容纳状态和外部可见性。`contained_by_id` 是容纳事实，不是方向目标；只有原文明示或必然推出时才填写。

`timeline_event_id` 引用承载同一动作阶段的事件。`narrative_required=true` 仅表示删除该阶段会丢失用户明确叙事或必然语义；不得借此增添新动作。

## 5. 时间轴

- `timeline.duration_seconds` 是整个视频片段的确定时长。原文只给范围时同时保留 `duration_range_seconds`；原文未给时长时使用当前契约的 15 秒结构缺省，标为 `default` 并登记 `use_default`，不得按动作数量推算。
- `timeline.events` 是可位于视频片段内任意数值起止点的语义事件，不限于开始、中间或结束三个位置。
- 各场景实体的动作阶段可以不同步、重叠、相接或留有间隔。不得按动作数量机械等分时间，不得因另一个实体开始动作就结束当前状态。
- 事件和动作阶段的起止时间必须位于总时长内，且开始时间不得晚于结束时间。

时间关系严格按两个事件的数值区间解释。设 source 为 `S`，target 为 `T`：

- `before`：S 的结束点不晚于 T 的开始点。
- `after`：S 的开始点不早于 T 的结束点。
- `meets`：S 的结束点等于 T 的开始点。
- `overlaps`：S 与 T 有非零重叠；该谓词不额外声明谁包含谁或端点是否相同。
- `during`：S 完整位于 T 的范围内。
- `starts_before`：S 的开始点不晚于 T；仅表达开始点次序，不声称结束点次序。
- `starts_after`：S 的开始点不早于 T；仅表达开始点次序，不声称结束点次序。
- `starts_with`：S 与 T 的开始点相同。
- `ends_with`：S 与 T 的结束点相同。

关系枚举与数值区间必须同时成立。只有原文明示可度量间隔时才填写 `minimum_gap_seconds` 或 `maximum_gap_seconds`；否则为 `null`。不得为通过校验而修改原文明示的时间关系；无法同时满足时登记不确定性。

## 6. 双实体空间关系

`scene_design.relationships` 只记录不能由单个动作阶段完整表达的双实体空间事实。允许的关系只有：

- 距离或尺度：`far_from`、`proximity`、`scale_dominance`
- 水平相对位置：`left_of`、`right_of`
- 纵深相对位置：`front_of`、`behind`
- 垂直相对位置：`above`、`below`

不得增加自由文本强度，不得输出同义或反向重复关系。被承载由 `motion_mode=carried + carrier_id` 表达；围绕目标的路径由 `relative_to_target + target_id + circular/elliptical` 表达；二者均不在空间关系数组中重复。

空间关系可以发生在任意时间：

- 全视频片段持续成立：`timeline_event_id=null` 且 `temporal_mode=throughout`。
- 某个任意时间区间持续成立：创建具有实际数值范围的事件，引用其 ID，并用 `throughout`。
- 只在事件开始或结束的边界成立：分别使用 `at_start` 或 `at_end`。

不得把只在局部时间成立的关系扩张为全片关系。

## 7. 摄影机、光学属性与画面构图

`camera` 只描述摄影机，`subject_motion` 只描述场景实体。两者的运动不得互相代填。

- `camera.movement.type` 表示摄影机自身的运动或旋转。固定位置改变朝向属于 `pan`；摄影机位置随目标一起改变属于 `follow`。只有需要目标的运镜才填写 `camera.movement.target_id`。
- `camera.focus_target_id` 表示取景关注对象，不等于运镜目标，也不等于动作目标。
- `camera.view_relation_to_motion` 只描述摄影机相对于主要线性运动的观察方位：`front/rear/side/three_quarter`。原文没有明确方位时必须用 `unspecified + default`。
- `camera.lens_intent` 只记录光学意图；不得把景别、画面占比或摄影机路径写入此字段。
- `composition.shot_size`、`screen_placements`、`visual_scales` 和 `visibility_requirements` 只记录原文明确的画面要求。
- 当前 Schema 的 `camera.movement` 只容纳一个连续主动运镜区间；原文若要求多个互不连续或互相矛盾的运镜阶段，不得静默合并，必须在 `uncertainties` 中说明结构限制。

`mood` 只保留原文的情绪、颜色、照明和氛围意图。不得由情绪词自行推导摄影机位置、运镜、景别或画面占比；应用中的确定性量化会处理允许的缺省映射。

## 8. 输出前强制审查

输出完整 JSON 前逐项确认：

1. 原文中每个有语义作用的场景实体、动作阶段、时间要求、空间关系、摄影机要求、光学属性、构图要求和氛围要求均已保留。
2. 所有 ID 引用都存在且指向正确类别；所有事件和动作阶段时间都在总时长内。
3. `action_kind`、`motion_type`、`motion_mode`、方向、目标、载体、路径和后置状态互相一致。
4. 时间关系名称与事件数值范围一致；空间关系的时间作用域没有被扩大。
5. 每个非空结论都有准确来源；非必然解释没有冒充事实。
6. 没有 Schema 之外字段、枚举、故事专用语义或依赖下游从自由文本重新猜测的缺失类型。

发现问题时直接输出修正后的完整对象。无法由原文决定时保留未知并登记 `uncertainties`，不得猜测。
