# Hermes 服务端视频代表帧迁移规格

状态：实施中  
日期：2026-08-22

## 目标

把视频视觉预处理的权威实现从 Panel 浏览器入口迁入 Hermes
`video_analyze`。无论视频来自本地上传、HTTPS URL、Asset Library、历史附件、
`asset_id` 或 `output_id`，在 Runtime 完成引用解析后都经过同一套服务端分析链。

## 当前问题

- Panel 只为 Composer 本地上传的视频抽取代表帧；其他媒体引用绕过该逻辑。
- Panel 上传的代表帧与原视频只以普通附件提交，没有来源视频或时间码合同。
- Hermes `video_analyze` 不使用这些帧，而是把完整视频编码为 Base64
  `video_url` 发送给视觉模型。
- 同一视频因入口不同获得不同的 Runtime 能力，Panel 因而成为错误的能力权威层。

## 范围

### Hermes

- 保持工具名 `video_analyze` 和现有参数不变。
- 保持现有视频来源解析、Run 所有权校验和可选音频转写不变。
- 在服务端使用已有 ffmpeg/ffprobe 基础设施，按 15%、50%、85% 抽取三张
  有界 JPEG 代表帧，最长边不超过 1280 像素。
- 将带时间码的代表帧作为 `image_url` 内容发送给辅助视觉模型；不得再把完整
  视频作为 Base64 `video_url` 放入 LLM 请求。
- ffmpeg/ffprobe 缺失、视频没有有效时长或帧提取失败时明确失败，不做无提示降级。
- 成功结果披露所使用的采样时间，便于调用方理解覆盖范围。

### Panel

- 普通视频上传与资产选择保持不变。
- 删除 Composer 提交前的浏览器代表帧派生和相应附件预算规则。
- 删除只服务于该浏览器抽帧路径的测试和错误文案。

## 非目标

- 本次不新增、传播或修改 egress budget。
- 本次不修改 Orchestrator、aiproxy、计费、审批或 provider 模型绑定。
- 本次不引入镜头检测、长视频分段、帧缓存或新的 Tool 参数。
- 本次不把 Skill 或 Orchestrator 改造成视频工作流执行器。

## 失败合同

- 媒体引用解析失败继续使用现有 Runtime 结构化错误。
- 服务端解码或抽帧失败返回非静默的 `video_analysis_failed`。
- 相同非重试失败的 Run 内重复调用继续由现有 Hermes Runtime guard 阻止。
- provider 是否开始提交仍由现有调用边界决定；抽帧失败发生在 provider 调用前。

## 验收

1. `video_analyze` 发给 LLM 的内容包含三张 `image_url`，不包含 `video_url`。
2. 每张代表帧携带可读时间码，顺序对应 15%、50%、85%。
3. 本地路径、HTTPS 下载及解析后的 `asset_id`/`output_id` 均复用同一函数。
4. ffmpeg/ffprobe 不可用或没有有效视频流时明确失败，且不调用 LLM。
5. Panel 视频提交不再生成 `.frame-N.jpg` 附件。
6. Hermes 视频与 Runtime 媒体相关测试、真实 ffmpeg 烟测和 Panel
   `npm run verify` 通过；无关 Hermes 全量套件留给 CI。
