# Cinematic Brief 语义契约 v0.21

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

### 1.1 Schema 字段对照

下列字段组就是 Structured Output Schema 提供的完整语义接口；不得把一个组的含义写入另一个组，也不得只填人类摘要而漏掉机器字段：

- 顶层对象只含 `summary/subjects/subject_motion/scene_dynamics/scene_design/mood/composition/camera/timeline/uncertainties`；`summary` 只做人类可读概括，不能代替任一结构字段。
- `subjects[]`：`id/category/description/narrative_role/attributes[]`；属性使用 `name/value/source_status/source_text`。
- `subject_motion[]`：`motion_id/subject_id/action/motion_semantics/direction/speed/trajectory/secondary_motion`；`secondary_motion` 当前必须为空。动作阶段的起止秒数只在 `timeline.events[]` 填写一次，动作通过 `motion_semantics.timeline_event_id` 引用它；Core 会在规范化 Brief 中确定性派生动作的 `start_time_seconds/end_time_seconds`。
- `motion_semantics`：`action_kind/motion_type/motion_mode/direction_mode/target_id/carrier_id/path_type/local_components/timeline_event_id/narrative_required/postconditions/source_status/source_text`；`postconditions` 只含 `contained_by_id/external_visibility`。
- `scene_dynamics`：`mode/source_status/reason`。
- `scene_design`：`environment/spatial_layers/relationships/environmental_motion`；`spatial_layers[]` 使用 `layer/content_ids/source_status/source_text`，`relationships[]` 使用 `type/subject_id/reference_id/timeline_event_id/temporal_mode/source_status/source_text`。
- `mood`：`emotional_tones/color_intent/lighting_intent/atmosphere`。这些字段只记录主观语义，不得改写主体运动或摄影机几何。
- `composition`：`shot_size/patterns/screen_placements/visual_scales/visibility_requirements`；画面位置使用 `subject_id/horizontal/vertical`，视觉比例使用 `subject_id/scale`，可见要求使用 `subject_id/requirement`。
- `camera`：`shot_type/view_angle/view_relation_to_motion/camera_height/lens_intent/focus_target_id/movement`；`movement` 使用 `type/target_id/direction/speed/trajectory/easing/start_time_seconds/end_time_seconds`。
- `timeline`：`duration_seconds/duration_range_seconds/duration_source_status/events/relations`；事件使用 `id/description/start_time_seconds/end_time_seconds/reference_ids/source_status/source_text`；事件间关系使用 `relation_id/source_event_id/target_event_id/relation/minimum_gap_seconds/maximum_gap_seconds/source_status/source_text`。
- `uncertainties[]`：`field/reason/resolution/selected_value`。它记录尚不能确定或按规则解决的具体字段，不替代任何已经能够确定的实体、事件或共有语义。

所有 `value/source_status/source_text` 注解对象共享第 2 节的来源规则。字段的类型、枚举、必填性和条件组合以同时发送的 JSON Schema 为准；本契约负责解释这些字段各自代表什么以及跨字段如何组合，二者不允许互相矛盾。

## 2. 原文证据与不确定性

所有带来源字段的值必须遵循：

- `explicit`：原文直接表达该值；`source_text` 保存支持它的最小原文片段。
- `inferred`：该值能由原文事实必然推出，而不是常见理解、较高概率或创作偏好；必须保留证据片段。
- `default`：本契约或 Schema 明确要求的结构缺省；`source_text` 为 `null`。
- `unknown`：原文不足以决定；值允许为空时应保持为空，`source_text` 为 `null`。

会影响几何、时间、方向、目标、路径、构图或摄影机行为的多种合理解释，不得擅自选择。将缺失点写入 `uncertainties`，并严格遵守 `[SEM-UNCERTAINTY-RESOLUTION]`：未选择时用 `unresolved + selected_value=null`；应用规则明确给出缺省时用 `use_default + 非空 selected_value`；只有必然推断时用 `use_inference + 非空 selected_value`。后两者的 `selected_value` 必须准确记录实际采用的字段值，不能只写理由或保留 null。摘要不得把推断、缺省或未知内容改写成用户原话。

不确定性必须拆解到真正存在分歧的字段。先列出所有与原文相容的合理解释，再把它们共有且能由当前 Schema 表达的实体参与、动作、局部事件或空间关系写入类型字段；只有方向、先后、路径形状、精确时刻等随解释变化的部分才保持未知。不得因为一个复合表述存在多种实现，就把各解释共有的可验证事实也从 Brief 删除。

用户明确表达的要求优先。不得为了填满字段，复制无关原文作为证据；不得从实体名称、单个汉字、语言子串或故事常识推出方向、目标、载体或时间关系。

## 3. 场景实体与环境

- 为每个需要独立定位、运动、构图或参与关系的对象创建一个稳定且唯一的 `subjects[].id`。
- 一个地点名词若只表示环境，写入 `scene_design.environment`；若其中对象还需要独立定位、运动、构图或参与关系，也必须创建场景实体。
- 只保留原文给出的类别、描述、叙事作用和属性。不能确定的属性不补写。
- `spatial_layers` 表达原文明确的前景、中景、后景等画面纵深分层，并引用已有场景实体。
- `environmental_motion` 只描述环境本身的动态，不代替场景实体的动作阶段。

`scene_dynamics` 只判断场景实体的可观察状态是否改变：位置、姿态、尺度、可见性或容纳状态在视频片段内发生任何变化即为 `dynamic`；全部保持不变即为 `static`。摄影机运动、焦距变化和纯环境效果不改变该分类。分类必须与所有动作阶段的类型字段一致。

## 4. 动作阶段与类型字段

每个场景实体独立建立完整的状态时间线。一个动作或状态具有独立起止范围时，必须建立独立的 `subject_motion` 和独立的 `timeline.events[]` 事件，并让 `motion_semantics.timeline_event_id` 引用该事件。只在事件上填写数值起止时间；`subject_motion` 不得重复输出起止秒数。动作阶段是语义区间，不是逐帧动画关键帧。允许不同动作通道重叠，但同一实体所有被引用事件区间的并集必须从 `0` 连续覆盖到 `timeline.duration_seconds`；这里要求的是**状态有解释**，绝不表示实体必须从 0 秒开始可见或运动。中途进入画面的实体应在进入前使用覆盖 `0..进入时刻` 的 hidden/尚未出现状态，并在该阶段末填写 `becomes_visible`；Planning 会据第一个显隐转变反推初始隐藏。没有这种明确状态时，执行层只能把空窗解释成“保持上一位置且继续可见”，会制造无依据停顿。

`action.value` 仅用于人类阅读和原文审计。以下字段共同构成下游唯一使用的机器语义，必须互相一致：

### 4.1 动作与驱动方式

- `motion_mode=stationary`：没有整体位移或局部 Transform 变化；必须搭配 `action_kind=hold`、`motion_type=static`、`path_type=stationary`、`local_components=[]`。显隐或容纳转变若确实发生，仍由独立 postcondition 表达并使 `scene_dynamics=dynamic`。
- `motion_mode=local_interaction`：只有局部姿态或尺度变化；必须搭配 `action_kind=interact`、`motion_type=interactive`、`path_type=stationary`，并在 `local_components` 中准确列出 `rotation` 和/或 `scale`。不得在不知道变化通道时同时猜两个。
- `motion_mode=self_propelled`：场景实体自主产生整体位移；必须搭配 `action_kind=locomotion`，`motion_type` 从 `walking/running/flying/jumping/moving` 中选择，路径不得为 `stationary`。
- `motion_mode=carried`：场景实体的世界运动由另一个场景实体承载；必须搭配 `action_kind=locomotion`、`motion_type=carried`、非空 `carrier_id`、`direction_mode=none`、`target_id=null`、`path_type=stationary`。该阶段不重复生成自主世界路径，但载体必须在 carried 的完整区间内具有 `self_propelled` 世界位移阶段；“被静止物体容纳”不是 carried。

`action_kind=other` 是 Schema 保留值，当前输出不得使用；无法分类时必须在 `uncertainties` 中保留问题，而不是绕过类型字段的一致性。`carrier_id` 仅在 `carried` 时填写，且必须引用另一个已有场景实体。

### 4.2 方向、目标与路径

- `direction_mode=none`：原文没有可执行的几何方向；`target_id` 必须为 `null`。
- `direction_mode=world_forward/world_left/world_right`：原文明确要求沿规范世界 `-Y/-X/+X`；`target_id` 必须为 `null`。这些值不是屏幕方向。
- `direction_mode=toward_target`：明确朝某场景实体接近；必须填写该实体的 `target_id`。
- `direction_mode=away_from_target`：明确背离某场景实体；必须填写该实体的 `target_id`。
- `direction_mode=relative_to_target`：明确以某场景实体为相对参考执行路径；必须填写该实体的 `target_id`。

动作的参与者、受影响对象、叙事关注对象和几何运动目标不是同一概念。仅当原文明确提供几何关系时才能填写非 `none` 方向和目标。普通的出现、经过、到达或离开只证明状态或位置发生变化，不自动证明相对于哪个场景实体运动。

`path_type` 只在原文提供足够证据时选用 `linear/circular/elliptical/s_curve/figure_eight/parabolic`；自主位移但路径形状未定时用 `unspecified`。不得发明某个故事专用动作类型或关系类型。

`postconditions` 只描述该动作阶段造成的容纳结果和外部可见性**转变**。`external_visibility=becomes_visible/becomes_hidden` 表示切换发生在该阶段末端；第一个转变若为 `becomes_visible`，则该实体在转变前初始隐藏，第一个转变若为 `becomes_hidden`，则转变前初始可见。一个场景实体从头到尾可见必须写 `unchanged`，绝不能因为结束时可见就写 `becomes_visible`。`contained_by_id` 是该阶段建立的容纳事实，不是方向目标；只有原文明示或必然推出时才填写。

`timeline_event_id` 必须引用承载同一动作阶段的事件，不允许为 `null`。`narrative_required=true` 仅表示删除该阶段会丢失用户明确叙事或必然语义；不得借此增添新动作。

## 5. 时间轴

- `timeline.duration_seconds` 是整个视频片段的确定时长。原文只给范围时同时保留 `duration_range_seconds`；原文未给时长时使用当前契约的 10 秒结构缺省，标为 `default` 并登记 `use_default`，不得按动作数量推算。
- `timeline.events` 是可位于视频片段内任意数值起止点的语义事件，不限于开始、中间或结束三个位置。
- 各场景实体的动作阶段可以不同步、重叠或相接，但每个实体自身的状态区间并集不得留有间隔。需要等待、停止、尚未出现或已经离场时，必须用与原文证据相符的状态阶段明确覆盖；不得靠省略阶段表达。不得按动作数量机械等分时间，也不得因另一个实体开始动作就结束当前状态。
- 原文没有表达“稍后、过一会、先停着、随后才出现”等延迟时，不得在第一个叙事动作之前自行制造静止或不可见空窗；第一个叙事动作从视频片段开始。原文没有表达动作提前结束并保持某状态时，最后一个叙事状态持续到视频片段结束。内部事件边界仍可根据事件顺序选择，但必须登记为推断或缺省，不能伪装成原文明示秒数。
- 复合载运叙事必须逐个列出所有实际改变世界位置的实体。若 B 在区间内 `carried` by A，B 只声明相对 A 不自主运动；A 仍必须拥有覆盖同一区间的 `self_propelled` 动作。不得因为“B 被 A 带走”只写 B 的 carried 而遗漏 A 的离开运动。
- 事件的起止时间必须位于总时长内，且必须严格满足开始时间早于结束时间。每个动作必须引用一个有效事件，Core 会从该事件派生动作区间；`timeline.events[].reference_ids` 只能引用已声明的场景实体，并列出该事件实际涉及的全部实体。
- 持续动作与发生在其内部的局部条件是两个不同事件。局部条件不得直接复用整个持续动作的时间范围。若关系只描述一个临界点、状态切换点或最近时刻而没有持续含义，使用位于持续动作内部的独立事件并配合 `at_midpoint`；只有原文明示在事件开始或结束边界成立时才分别使用 `at_start/at_end`。若关系确实持续一段时间，才使用严格更短的有界区间与 `throughout`。原文未给精确时刻或区间时，采用上述结构缺省并登记时间不确定性，不得伪装成原文明示时间。

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

`scene_design.relationships` 只记录不能由单个动作阶段完整表达的双实体空间事实。每条都按“`subject_id` 满足谓词，`reference_id` 是参照物”读取。允许的关系只有：

- `far_from`：subject 在当前摄影机的纵深上比 reference 更远，并与 reference 保有世界空间净空。例如“男人面前远处有飞船”必须是 `subject_id=飞船`、`reference_id=男人`。
- `proximity`：subject 与 reference 在世界空间接近；该关系对交换两者语义对称，但仍只输出一个规范记录。
- `scale_dominance`：subject 在**最终画面投影**中显著大于 reference。单纯说某物“巨大/很大”只描述物理尺寸，必须写入该实体的描述或属性，不能据此输出 `scale_dominance`。
- 水平相对位置：`left_of`、`right_of`
- 纵深相对位置：`front_of`、`behind`
- 垂直相对位置：`above`、`below`

`left_of/right_of/front_of/behind/above/below` 使用规范世界坐标，而不是屏幕方向或摄影机视角：左/右是 `-X/+X`，前/后是 `-Y/+Y`，上/下是 `+Z/-Z`。若原文只表达“画面左侧/右侧”，应写入 `composition.screen_placements`，不得写世界关系。

不得增加自由文本强度，不得输出同义或反向重复关系。被承载由 `motion_mode=carried + carrier_id` 表达；围绕目标的路径由 `relative_to_target + target_id + circular/elliptical` 表达；二者均不在空间关系数组中重复。

空间关系可以发生在任意时间：

- 全视频片段持续成立：`timeline_event_id=null` 且 `temporal_mode=throughout`。
- 某个任意时间区间持续成立：创建具有实际数值范围的事件，引用其 ID，并用 `throughout`。
- 只在事件中的一个临界时刻成立且原文未指定边界：使用 `at_midpoint`；原文明示开始或结束边界时才分别使用 `at_start` 或 `at_end`。

不得把只在局部时间成立的关系扩张为全片关系。

当一个多实体表述存在不同运动方向或路径解释，但所有合理解释都要求同一局部空间关系时，必须保留这条共有关系，并为它创建独立的局部事件；方向和路径继续保持未知。关系描述的是需要验证的语义事实，动作路径点只是下游实现手段，两者不得互相替代。

## 7. 摄影机、光学属性与画面构图

`camera` 只描述摄影机，`subject_motion` 只描述场景实体。两者的运动不得互相代填。

- `camera.movement.type` 是代码直接执行的封闭枚举：`static/pan/orbit/push_in/pull_out/follow/lateral`。固定位置改变朝向属于 `pan`；摄影机位置随目标一起改变属于 `follow`。只有 `pan/follow/orbit` 填写 `camera.movement.target_id`，其他类型必须为 `null`。不得输出“缓慢推近”等自由文本类型；速度单独写入 `speed`。
- `camera.focus_target_id` 表示取景关注对象，不等于运镜目标，也不等于动作目标。
- `camera.view_relation_to_motion` 只描述摄影机相对于主要线性运动的观察方位：`front/rear/side/three_quarter`。原文没有明确方位时必须用 `unspecified + default`。
- `camera.lens_intent` 只记录光学意图；不得把景别、画面占比或摄影机路径写入此字段。
- 当前 `composition.screen_placements` 只接受水平 `left/center/right` 与垂直 `top/center/bottom`；`visual_scales.scale` 只接受百分比或百分比范围（如 `20%`、`20%-30%`）；`visibility_requirements.requirement` 只接受 `keep_in_frame`。这些类型化值由代码直接执行，不得写同义自然语言。
- `composition.shot_size/patterns`、`camera.shot_type`、`camera.movement.direction/trajectory/easing` 和 `subject_motion.secondary_motion` 尚无百分百对接的执行接口，当前必须保持 unknown/空数组。若原文只提供这些尚不可执行的要求，在 `uncertainties` 中记录结构限制，不能把它们伪装成已实现字段。
- 明确的 `camera.camera_height` 必须写成米制数值（如 `1.6 m`）；明确的 `camera.lens_intent` 必须是毫米焦距或 `wide/normal/telephoto`；明确的 `camera.view_angle` 必须是 `eye_level/high_angle/low_angle/top_down`。
- 当前 Schema 的 `camera.movement` 只容纳一个连续主动运镜区间；原文若要求多个互不连续或互相矛盾的运镜阶段，不得静默合并，必须在 `uncertainties` 中说明结构限制。

`mood` 只保留原文的情绪、颜色、照明和氛围意图。不得由情绪词自行推导摄影机位置、运镜、景别或画面占比；应用中的确定性量化会处理允许的缺省映射。

## 8. 输出前强制审查

输出完整 JSON 前逐项确认：

1. 原文中每个有语义作用的场景实体、动作阶段、时间要求、空间关系、摄影机要求、光学属性、构图要求和氛围要求均已保留。
2. 所有 ID 引用都存在且指向正确类别；所有事件和动作阶段时间都在总时长内。
3. `action_kind`、`motion_type`、`motion_mode`、方向、目标、载体、路径和后置状态互相一致。
4. 时间关系名称与事件数值范围一致；空间关系的时间作用域没有被扩大。
5. 每个非空结论都有准确来源；非必然解释没有冒充事实；每项不确定性都只覆盖真正分歧的字段，没有吞掉合理解释共有的类型化事实。
6. 局部关系拥有独立的点事件或严格短于其承载动作的区间事件，没有被扩张为整个动作或整个视频片段；`throughout` 只用于确有持续含义的关系。
7. 没有 Schema 之外字段、枚举、故事专用语义或依赖下游从自由文本重新猜测的缺失类型。
8. `scene_dynamics=dynamic` 至少对应一个非静止动作、明确的 `local_components`，或 `becomes_visible/becomes_hidden`/新建容纳关系；纯摄影机运动、静态可见和静态构图不能作为 dynamic 理由。
9. 每个场景实体的状态区间并集是否完整覆盖视频片段；是否存在无原文依据的可见静止空窗。
10. 每个 narrative-required carried 区间是否都有同一时间范围内持续移动的载体；“带走”是否同时为载体建立了离开阶段。
11. 按 `[SEM-UNCERTAINTY-RESOLUTION]` 逐项检查 `uncertainties`；特别禁止 `use_default/use_inference + selected_value=null` 和 `unresolved + 非空 selected_value`。

发现问题时直接输出修正后的完整对象。无法由原文决定时保留未知并登记 `uncertainties`，不得猜测。
