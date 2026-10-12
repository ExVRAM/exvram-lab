const invoke = (...args) => window.__TAURI__.core.invoke(...args);

const panel = document.querySelector("#panel");
const notice = document.querySelector("#notice");
let view = "panel";
let latest = null;
let timer = null;

function text(value) {
  return value == null || value === "" ? "Нет данных" : String(value);
}

function row(label, value) {
  return `<div class="row"><span>${label}</span><strong>${text(value)}</strong></div>`;
}

function section(title, body) {
  return `<section><h2>${title}</h2>${body}</section>`;
}

function render(data) {
  latest = data;
  notice.textContent = data.notice || "";
  notice.className = data.notice && /отмен|не |ошиб|больше|нестабил/i.test(data.notice) ? "notice warn" : "notice";
  if (view === "report") {
    panel.innerHTML = [
      section("Current configuration", [
        row("Контекст", data.configuredContext),
        row("KV", data.kv),
        row("Режим", data.mode),
        row("Protect My PC", data.protect ? "включён" : "выключен"),
      ].join("")),
      section("Recommended configuration", row("Рецепт", data.recommendation)),
      section("Expected memory change", row("VRAM", data.memoryChange)),
      section("Expected speed change", row("Decode", data.speedChange)),
      section("Quality risk", row("Оценка", data.qualityRisk)),
      section("Restart required", row("Перезапуск", data.restartRequired ? "да, при следующем запуске" : "нет")),
    ].join("");
    return;
  }
  const presets = (data.presets || []).map((item) => {
    const pressed = String(data.configuredContext || "").startsWith(String(item.tokens));
    return `<button type="button" data-context="${item.tokens}" aria-pressed="${pressed}" ${item.allowed ? "" : "disabled"} title="${text(item.estimate)}">${item.label}</button>`;
  }).join("");
  const models = (data.models || []).slice(0, 8).map((item) => {
    return `<button type="button" data-model="${encodeURIComponent(item.path)}">${item.name}<small>${item.verified ? "проверенный рецепт" : "нет сравнительных измерений"}</small></button>`;
  }).join("");
  panel.innerHTML = [
    section("Модель", `<div class="grid">
      <div class="cell"><span>Название</span><strong>${text(data.modelName)}</strong></div>
      <div class="cell"><span>Параметры</span><strong>${text(data.parameters)}</strong></div>
      <div class="cell"><span>Квантование</span><strong>${text(data.quant)}</strong></div>
      <div class="cell"><span>Backend</span><strong>${text(data.backend)}</strong></div>
    </div>${row("Состояние", data.loadState)}
    <div class="actions" style="margin-top:8px">
      <button class="primary" type="button" data-cmd="start">Запустить</button>
      <button type="button" data-cmd="stop">Остановить</button>
      <button type="button" data-cmd="pick_model">Файл</button>
    </div>`),
    section("Видеокарта", [
      row("GPU", data.gpuName),
      row("Всего VRAM", data.vramTotal),
      row("Занято", data.vramUsed),
      row("Свободно", data.vramFree),
      row("Температура", data.temperature),
      row("Загрузка", data.gpuLoad),
    ].join("")),
    section("Производительность", [
      row("Decode", data.decode),
      row("Prefill", data.prefill),
      row("TTFT", data.ttft),
      row("CPU", data.cpu),
      row("RAM", data.ram),
    ].join("")),
    section("Контекст", [
      row("Configured", data.configuredContext),
      row("Actual occupancy", data.occupiedContext),
      row("Свободно", data.freeContext),
      row("KV", data.kv),
      row("Память KV", data.kvMemory),
    ].join("") + `<div class="presets" style="margin-top:8px">${presets}</div>
      <label>Custom<input id="custom-context" inputmode="numeric" placeholder="токены"></label>`),
    section("Auto Optimize", `<div class="choices">
      ${["balanced", "performance", "quality"].map((mode) => `<button type="button" data-mode="${mode}" aria-pressed="${data.mode === mode}">${mode}</button>`).join("")}
    </div>
    <label class="row"><span>Protect My PC</span><input id="protect" type="checkbox" ${data.protect ? "checked" : ""}></label>`),
    section("Профили", `<div class="choices">
      ${["chat", "documents", "coding", "research", "custom"].map((profile) => `<button type="button" data-profile="${profile}" aria-pressed="${data.profile === profile}">${profile}</button>`).join("")}
    </div>`),
    section("Context Optimization", `<p class="notice">OFF — единственный рабочий режим. AUTO и EXPERIMENTAL не подключены: TurboQuant, KVQuant, KIVI, SnapKV, PyramidKV и LLMLingua в этой сборке нет. KV q4/q8/f16 — реальные флаги llama.cpp, не сжатие промпта.</p>
      <div class="choices">
        <button type="button" data-kv="q4_0" aria-pressed="${data.kv === "q4_0"}">q4_0</button>
        <button type="button" data-kv="q8_0" aria-pressed="${data.kv === "q8_0"}">q8_0</button>
        <button type="button" data-kv="f16" aria-pressed="${data.kv === "f16"}">f16</button>
      </div>`),
    section("Модели на диске", `<div class="models">${models || "<p class=\"notice\">Нет данных</p>"}</div>
      <div class="actions"><button type="button" data-cmd="pick_models_dir">Папка</button><button type="button" data-cmd="pick_server">llama-server</button></div>
      ${row("Сервер", data.serverLabel)}`),
    section("Advanced", `
      ${row("GPU offload", "999")}
      ${row("logging", "verbosity 3")}
      <label>batch<input id="batch" value="${data.batch}"></label>
      <label>ubatch<input id="ubatch" value="${data.ubatch}"></label>
      <label class="row"><span>Flash Attention</span><input id="flash" type="checkbox" ${data.flashAttn ? "checked" : ""}></label>
      <label>threads, пусто = авто<input id="threads" value="${data.threads === "авто" ? "" : text(data.threads)}"></label>
      <div class="actions" style="margin-top:8px">
        <button type="button" data-cmd="advanced">Применить</button>
        <button type="button" data-cmd="reset_recommended">Reset to Recommended</button>
        <button type="button" data-cmd="export_settings">Экспорт</button>
        <button type="button" data-cmd="import_settings">Импорт</button>
      </div>`),
  ].join("");
}

async function call(name, args) {
  try {
    const data = await invoke(name, args || {});
    if (data) render(data);
  } catch (error) {
    notice.textContent = String(error);
    notice.className = "notice warn";
  }
}

panel.addEventListener("click", (event) => {
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.cmd === "advanced") {
    const threads = document.querySelector("#threads").value.trim();
    const payload = {
      batch: Number(document.querySelector("#batch").value),
      ubatch: Number(document.querySelector("#ubatch").value),
      flashAttn: document.querySelector("#flash").checked,
      threads: threads ? Number(threads) : null,
      acknowledged: false,
    };
    invoke("apply_advanced", payload).then(render).catch((error) => {
      const message = typeof error === "string" ? error : (error && error.message) || String(error);
      if (message.includes("Проверьте") && window.confirm(message)) {
        call("apply_advanced", { ...payload, acknowledged: true });
      } else {
        notice.textContent = message;
        notice.className = "notice warn";
      }
    });
    return;
  }
  if (target.dataset.cmd) call(target.dataset.cmd);
  if (target.dataset.context) call("set_context", { tokens: Number(target.dataset.context) });
  if (target.dataset.kv) call("set_kv", { kv: target.dataset.kv });
  if (target.dataset.mode) call("set_mode", { mode: target.dataset.mode });
  if (target.dataset.profile) call("set_profile", { profile: target.dataset.profile });
  if (target.dataset.model) call("select_listed_model", { path: decodeURIComponent(target.dataset.model) });
});

panel.addEventListener("change", (event) => {
  if (event.target.id === "protect") call("set_protect", { enabled: event.target.checked });
  if (event.target.id === "custom-context") {
    const tokens = Number(event.target.value);
    if (tokens) call("set_context", { tokens });
  }
});

document.querySelector("#view-panel").onclick = () => { view = "panel"; document.querySelector("#view-panel").classList.add("active"); document.querySelector("#view-report").classList.remove("active"); if (latest) render(latest); };
document.querySelector("#view-report").onclick = () => { view = "report"; document.querySelector("#view-report").classList.add("active"); document.querySelector("#view-panel").classList.remove("active"); if (latest) render(latest); };

let away = false;

function schedule() {
  clearInterval(timer);
  const hidden = away || document.visibilityState === "hidden";
  timer = setInterval(() => call("snapshot"), hidden ? 8000 : 2000);
}

document.addEventListener("visibilitychange", schedule);

if (!window.__TAURI__) {
  notice.textContent = "Окно открыто вне ExVRAM Control.";
} else {
  window.__TAURI__.event.listen("presence", (event) => {
    away = event.payload === "tray";
    schedule();
  });
  call("snapshot");
  schedule();
}
