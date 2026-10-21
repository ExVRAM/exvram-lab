#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <tlhelp32.h>
#include <winhttp.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kLoadTimeoutMs = 120000;
constexpr wchar_t kRuntimeRelative[] =
    L".cache\\llama\\runtime-b10982-cuda13.4\\llama-server.exe";
constexpr wchar_t kModelRelative[] = L"models\\qwen35-2b\\Qwen3.5-2B-Q4_K_M.gguf";
constexpr int kDefaultPredict = 256;
constexpr int kDefaultContext = 8448;
constexpr int kDefaultBatch = 32;
constexpr int kDefaultUbatch = 1;

struct CudaDevice {
  std::wstring name;
  int major = 0;
  int minor = 0;
  int multiprocessors = 0;
};

std::wstring quote_arg(const std::wstring& arg) {
  if (arg.empty()) {
    return L"\"\"";
  }
  if (arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    return arg;
  }
  std::wstring out = L"\"";
  size_t i = 0;
  while (i < arg.size()) {
    size_t slashes = 0;
    while (i < arg.size() && arg[i] == L'\\') {
      ++slashes;
      ++i;
    }
    if (i == arg.size()) {
      out.append(slashes * 2, L'\\');
      break;
    }
    if (arg[i] == L'"') {
      out.append(slashes * 2 + 1, L'\\');
      out.push_back(L'"');
    } else {
      out.append(slashes, L'\\');
      out.push_back(arg[i]);
    }
    ++i;
  }
  out.push_back(L'"');
  return out;
}

std::wstring join_command(const std::vector<std::wstring>& args) {
  std::wstring line;
  for (const auto& arg : args) {
    if (!line.empty()) {
      line.push_back(L' ');
    }
    line += quote_arg(arg);
  }
  return line;
}

std::filesystem::path executable_dir() {
  std::wstring buffer(32768, L'\0');
  DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  buffer.resize(length);
  return std::filesystem::path(buffer).parent_path();
}

std::filesystem::path find_root() {
  std::vector<std::filesystem::path> starts = {std::filesystem::current_path(), executable_dir()};
  for (auto dir : starts) {
    for (int hop = 0; hop < 8 && !dir.empty(); ++hop) {
      if (std::filesystem::exists(dir / kRuntimeRelative) &&
          std::filesystem::exists(dir / kModelRelative)) {
        return dir;
      }
      auto parent = dir.parent_path();
      if (parent == dir) {
        break;
      }
      dir = parent;
    }
  }
  return {};
}

using CuInit = int(__stdcall*)(unsigned);
using CuDeviceGetCount = int(__stdcall*)(int*);
using CuDeviceGet = int(__stdcall*)(int*, int);
using CuDeviceGetName = int(__stdcall*)(char*, int, int);
using CuDeviceGetAttribute = int(__stdcall*)(int*, int, int);

CudaDevice query_cuda_device() {
  HMODULE library = LoadLibraryW(L"nvcuda.dll");
  if (library == nullptr) {
    throw std::runtime_error("nvcuda.dll is not loaded. The NVIDIA driver is missing.");
  }
  auto init = reinterpret_cast<CuInit>(GetProcAddress(library, "cuInit"));
  auto count_fn = reinterpret_cast<CuDeviceGetCount>(GetProcAddress(library, "cuDeviceGetCount"));
  auto get = reinterpret_cast<CuDeviceGet>(GetProcAddress(library, "cuDeviceGet"));
  auto name_fn = reinterpret_cast<CuDeviceGetName>(GetProcAddress(library, "cuDeviceGetName"));
  auto attr = reinterpret_cast<CuDeviceGetAttribute>(GetProcAddress(library, "cuDeviceGetAttribute"));
  if (init == nullptr || count_fn == nullptr || get == nullptr || name_fn == nullptr || attr == nullptr) {
    throw std::runtime_error("nvcuda.dll does not export the driver entry points.");
  }
  if (init(0) != 0) {
    throw std::runtime_error("cuInit failed.");
  }
  int count = 0;
  if (count_fn(&count) != 0 || count < 1) {
    throw std::runtime_error("The CUDA driver reports no device.");
  }
  int device = 0;
  if (get(&device, 0) != 0) {
    throw std::runtime_error("cuDeviceGet failed.");
  }
  char name[256] = {};
  if (name_fn(name, sizeof(name), device) != 0) {
    throw std::runtime_error("cuDeviceGetName failed.");
  }
  CudaDevice info;
  info.name.assign(name, name + std::strlen(name));
  constexpr int kMajor = 75;
  constexpr int kMinor = 76;
  constexpr int kMultiprocessors = 16;
  if (attr(&info.major, kMajor, device) != 0 || attr(&info.minor, kMinor, device) != 0 ||
      attr(&info.multiprocessors, kMultiprocessors, device) != 0) {
    throw std::runtime_error("cuDeviceGetAttribute failed.");
  }
  return info;
}

std::string narrow(const std::wstring& text) {
  if (text.empty()) {
    return {};
  }
  int size = WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
  std::string out(size, '\0');
  WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), size, nullptr, nullptr);
  return out;
}

bool module_loaded(DWORD pid, const wchar_t* name) {
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid);
  if (snapshot == INVALID_HANDLE_VALUE) {
    return false;
  }
  MODULEENTRY32W entry{};
  entry.dwSize = sizeof(entry);
  bool found = false;
  if (Module32FirstW(snapshot, &entry)) {
    do {
      if (_wcsicmp(entry.szModule, name) == 0) {
        found = true;
        break;
      }
    } while (Module32NextW(snapshot, &entry));
  }
  CloseHandle(snapshot);
  return found;
}

std::string run_capture(const std::wstring& command) {
  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  HANDLE read = nullptr;
  HANDLE write = nullptr;
  if (!CreatePipe(&read, &write, &security, 0)) {
    return {};
  }
  SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdOutput = write;
  startup.hStdError = write;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION process{};
  std::wstring mutable_command = command;
  BOOL ok = CreateProcessW(
      nullptr, mutable_command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr,
      &startup, &process);
  CloseHandle(write);
  std::string output;
  if (ok) {
    char buffer[4096];
    DWORD got = 0;
    while (ReadFile(read, buffer, sizeof(buffer), &got, nullptr) && got > 0) {
      output.append(buffer, buffer + got);
    }
    WaitForSingleObject(process.hProcess, 15000);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
  }
  CloseHandle(read);
  return output;
}

std::string llama_gpu_line(const std::string& snapshot) {
  std::string kept;
  size_t start = 0;
  while (start < snapshot.size()) {
    auto end = snapshot.find('\n', start);
    if (end == std::string::npos) {
      end = snapshot.size();
    }
    std::string line = snapshot.substr(start, end - start);
    if (line.find("llama-server.exe") != std::string::npos) {
      if (!kept.empty()) {
        kept.push_back('\n');
      }
      kept += line;
    }
    start = end + (end < snapshot.size() ? 1 : 0);
  }
  return kept;
}

std::string json_escape(const std::wstring& text) {
  std::string utf8 = narrow(text);
  std::string out;
  for (unsigned char c : utf8) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

std::string json_string_after(const std::string& body, const std::string& key) {
  auto at = body.find(key);
  if (at == std::string::npos) {
    return {};
  }
  at = body.find('"', at + key.size());
  if (at == std::string::npos) {
    return {};
  }
  ++at;
  std::string out;
  while (at < body.size()) {
    char c = body[at++];
    if (c == '\\' && at < body.size()) {
      char next = body[at++];
      out.push_back(next == 'n' ? '\n' : next);
    } else if (c == '"') {
      break;
    } else {
      out.push_back(c);
    }
  }
  return out;
}

INTERNET_PORT free_local_port() {
  WSADATA data{};
  if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
    throw std::runtime_error("WSAStartup failed.");
  }
  SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (sock == INVALID_SOCKET) {
    WSACleanup();
    throw std::runtime_error("socket failed.");
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    closesocket(sock);
    WSACleanup();
    throw std::runtime_error("Cannot reserve a local port.");
  }
  int size = sizeof(addr);
  if (getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &size) != 0) {
    closesocket(sock);
    WSACleanup();
    throw std::runtime_error("getsockname failed.");
  }
  INTERNET_PORT port = ntohs(addr.sin_port);
  closesocket(sock);
  WSACleanup();
  return port;
}

bool http_request(
    INTERNET_PORT port, const wchar_t* verb, const wchar_t* path, const std::string& body, DWORD* status,
    std::string* response) {
  HINTERNET session = WinHttpOpen(
      L"exvram-cuda/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (session == nullptr) {
    return false;
  }
  WinHttpSetTimeouts(session, 2000, 2000, 5000, 120000);
  HINTERNET connect = WinHttpConnect(session, L"127.0.0.1", port, 0);
  if (connect == nullptr) {
    WinHttpCloseHandle(session);
    return false;
  }
  HINTERNET request = WinHttpOpenRequest(
      connect, verb, path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
  if (request == nullptr) {
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);
    return false;
  }
  BOOL sent = WinHttpSendRequest(
      request, L"Content-Type: application/json\r\n", static_cast<DWORD>(-1),
      body.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(body.data()),
      static_cast<DWORD>(body.size()), static_cast<DWORD>(body.size()), 0);
  BOOL received = sent && WinHttpReceiveResponse(request, nullptr);
  if (!received) {
    WinHttpCloseHandle(request);
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);
    return false;
  }
  DWORD code = 0;
  DWORD code_size = sizeof(code);
  WinHttpQueryHeaders(
      request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code,
      &code_size, WINHTTP_NO_HEADER_INDEX);
  std::string payload;
  DWORD got = 0;
  do {
    char buffer[4096];
    if (!WinHttpReadData(request, buffer, sizeof(buffer), &got)) {
      break;
    }
    payload.append(buffer, buffer + got);
  } while (got > 0);
  WinHttpCloseHandle(request);
  WinHttpCloseHandle(connect);
  WinHttpCloseHandle(session);
  *status = code;
  *response = std::move(payload);
  return true;
}

bool full_gpu_offload(const std::string& log, int* done, int* total) {
  const std::string marker = "offloaded ";
  auto at = log.rfind(marker);
  if (at == std::string::npos) {
    return false;
  }
  at += marker.size();
  auto slash = log.find('/', at);
  auto end = log.find(" layers to GPU", at);
  if (slash == std::string::npos || end == std::string::npos || slash > end) {
    return false;
  }
  try {
    *done = std::stoi(log.substr(at, slash - at));
    *total = std::stoi(log.substr(slash + 1, end - slash - 1));
  } catch (const std::exception&) {
    return false;
  }
  return *done > 0 && *done == *total;
}

std::string field_after(const std::string& log, const std::string& key) {
  auto at = log.rfind(key);
  if (at == std::string::npos) {
    return {};
  }
  at += key.size();
  while (at < log.size() && (log[at] == ' ' || log[at] == '=')) {
    ++at;
  }
  auto end = log.find_first_of("\r\n", at);
  return log.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

class GpuServer {
 public:
  ~GpuServer() { stop(); }

  void start(
      const std::filesystem::path& runtime, const std::filesystem::path& model, INTERNET_PORT port, int context,
      int batch, int ubatch, bool embedding_gpu) {
    port_ = port;
    unsigned cores = std::thread::hardware_concurrency();
    int threads = cores > 2 ? static_cast<int>(cores) - 2 : 1;
    std::vector<std::wstring> args = {
        runtime.wstring(),
        L"--model",
        model.wstring(),
        L"--host",
        L"127.0.0.1",
        L"--port",
        std::to_wstring(port_),
        L"--no-webui",
        L"--ctx-size",
        std::to_wstring(context),
        L"--gpu-layers",
        L"999",
        L"--device",
        L"CUDA0",
        L"--split-mode",
        L"none",
        L"--fit",
        L"off",
        L"--load-mode",
        L"dio",
        L"--prio",
        L"-1",
        L"--threads",
        std::to_wstring(threads),
        L"--threads-batch",
        std::to_wstring(threads),
        L"--batch-size",
        std::to_wstring(batch),
        L"--ubatch-size",
        std::to_wstring(ubatch),
        L"--flash-attn",
        L"on",
        L"--cache-type-k",
        L"q4_0",
        L"--cache-type-v",
        L"q4_0",
        L"--parallel",
        L"1",
        L"--reasoning",
        L"off",
        L"--ctx-checkpoints",
        L"0",
        L"--cache-ram",
        L"0",
        L"--verbosity",
        L"4",
    };
    if (embedding_gpu) {
      args.push_back(L"--override-tensor");
      args.push_back(L"token_embd.weight=CUDA0");
    }

    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE write = nullptr;
    if (!CreatePipe(&read_, &write, &security, 0)) {
      throw std::runtime_error("CreatePipe failed.");
    }
    SetHandleInformation(read_, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ, &security, OPEN_EXISTING, 0, nullptr);
    if (nul == INVALID_HANDLE_VALUE) {
      CloseHandle(read_);
      CloseHandle(write);
      read_ = nullptr;
      throw std::runtime_error("Cannot open NUL for the model stdin.");
    }

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = nul;
    startup.hStdOutput = write;
    startup.hStdError = write;
    std::wstring command = join_command(args);
    BOOL created = CreateProcessW(
        runtime.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
        runtime.parent_path().c_str(), &startup, &process_);
    CloseHandle(write);
    CloseHandle(nul);
    if (!created) {
      CloseHandle(read_);
      read_ = nullptr;
      throw std::runtime_error("llama-server did not start. Error " + std::to_string(GetLastError()));
    }

    job_ = CreateJobObjectW(nullptr, nullptr);
    if (job_ != nullptr) {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
      limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
      AssignProcessToJobObject(job_, process_.hProcess);
    }

    reader_ = std::thread([this] { drain(); });
  }

  void wait_until_ready() {
    const DWORD deadline = GetTickCount() + kLoadTimeoutMs;
    while (GetTickCount() < deadline) {
      if (WaitForSingleObject(process_.hProcess, 0) == WAIT_OBJECT_0) {
        stop();
        throw std::runtime_error("llama-server exited before it accepted requests.");
      }
      DWORD status = 0;
      std::string body;
      if (http_request(port_, L"GET", L"/health", {}, &status, &body) && status == 200) {
        sample_gpu();
        return;
      }
      Sleep(200);
    }
    stop();
    throw std::runtime_error("llama-server did not become ready within 120 seconds.");
  }

  std::string ask(const std::wstring& prompt, int predict) {
    std::string body = "{\"temperature\":0,\"max_tokens\":" + std::to_string(predict) +
                       ",\"messages\":[{\"role\":\"user\",\"content\":\"" + json_escape(prompt) + "\"}]}";
    DWORD status = 0;
    std::string response;
    if (!http_request(port_, L"POST", L"/v1/chat/completions", body, &status, &response) || status != 200) {
      throw std::runtime_error("The completion request failed.");
    }
    std::string answer = json_string_after(response, "\"content\"");
    if (answer.empty()) {
      throw std::runtime_error("The completion response has no content.");
    }
    return answer;
  }

  std::string log() const {
    std::lock_guard<std::mutex> lock(guard_);
    return log_;
  }

  bool cuda_module_loaded() const { return cuda_module_; }

  const std::string& nvidia_line() const { return nvidia_line_; }

  INTERNET_PORT port() const { return port_; }

  void stop() {
    if (process_.hProcess != nullptr) {
      TerminateProcess(process_.hProcess, 0);
      WaitForSingleObject(process_.hProcess, 5000);
      CloseHandle(process_.hThread);
      CloseHandle(process_.hProcess);
      process_ = {};
    }
    if (reader_.joinable()) {
      reader_.join();
    }
    if (read_ != nullptr) {
      CloseHandle(read_);
      read_ = nullptr;
    }
    if (job_ != nullptr) {
      CloseHandle(job_);
      job_ = nullptr;
    }
  }

 private:
  void drain() {
    std::string local;
    char buffer[4096];
    DWORD got = 0;
    while (read_ != nullptr && ReadFile(read_, buffer, sizeof(buffer), &got, nullptr) && got > 0) {
      local.append(buffer, buffer + got);
      std::lock_guard<std::mutex> lock(guard_);
      log_ = local;
    }
  }

  void sample_gpu() {
    if (process_.dwProcessId == 0) {
      return;
    }
    cuda_module_ = module_loaded(process_.dwProcessId, L"ggml-cuda.dll");
    nvidia_line_ = llama_gpu_line(run_capture(
        L"nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader"));
  }

  INTERNET_PORT port_ = 0;
  PROCESS_INFORMATION process_{};
  HANDLE job_ = nullptr;
  HANDLE read_ = nullptr;
  std::thread reader_;
  mutable std::mutex guard_;
  std::string log_;
  std::string nvidia_line_;
  bool cuda_module_ = false;
};

void usage() {
  std::wcerr << L"exvram-cuda [--root DIR] [--runtime EXE] [--model PATH]\n"
             << L"            [--ctx-size N] [--batch-size N] [--ubatch-size N]\n"
             << L"            [--embedding-gpu on|off] [--prompt TEXT] [--predict N]\n"
             << L"Starts the ExVRAM llama-server recipe on CUDA0.\n"
             << L"Without --prompt, reads lines until /exit. Ctrl+C stops the server.\n";
}

struct Placement {
  int done = 0;
  int total = 0;
  std::string buffer;
  std::string splits;
  bool cuda_loaded = false;
  std::string nvidia;
};

Placement require_cuda(GpuServer& server, bool embedding_gpu) {
  Placement placement;
  std::string log = server.log();
  placement.cuda_loaded = server.cuda_module_loaded();
  placement.nvidia = server.nvidia_line();
  bool on_gpu = full_gpu_offload(log, &placement.done, &placement.total);
  placement.buffer = field_after(log, "CUDA0 model buffer size");
  placement.splits = field_after(log, "graph splits");
  std::cout << "gpu_layers: " << placement.done << "/" << placement.total << "\n";
  std::cout << "cuda_model_buffer: " << (placement.buffer.empty() ? "missing" : placement.buffer) << "\n";
  std::cout << "graph_splits: " << (placement.splits.empty() ? "missing" : placement.splits) << "\n";
  std::cout << "ggml_cuda_loaded: " << (placement.cuda_loaded ? "yes" : "no") << "\n";
  if (!placement.nvidia.empty()) {
    std::cout << "nvidia_smi: " << placement.nvidia << "\n";
  }
  bool whole_graph = !embedding_gpu || placement.splits == "1";
  if (!on_gpu || placement.buffer.empty() || !whole_graph) {
    server.stop();
    throw std::runtime_error("CUDA offload did not keep the decoder on CUDA0.");
  }
  return placement;
}

std::wstring widen_utf8(const std::string& bytes) {
  if (bytes.empty()) {
    return {};
  }
  int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, bytes.data(), static_cast<int>(bytes.size()), nullptr, 0);
  if (count <= 0) {
    count = MultiByteToWideChar(CP_ACP, 0, bytes.data(), static_cast<int>(bytes.size()), nullptr, 0);
    if (count <= 0) {
      return {};
    }
    std::wstring wide(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_ACP, 0, bytes.data(), static_cast<int>(bytes.size()), wide.data(), count);
    return wide;
  }
  std::wstring wide(static_cast<size_t>(count), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, bytes.data(), static_cast<int>(bytes.size()), wide.data(), count);
  return wide;
}

bool read_prompt_line(std::wstring* line) {
  HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode = 0;
  if (GetConsoleMode(input, &mode)) {
    wchar_t buffer[4096];
    DWORD got = 0;
    if (!ReadConsoleW(input, buffer, 4095, &got, nullptr)) {
      return false;
    }
    while (got > 0 && (buffer[got - 1] == L'\n' || buffer[got - 1] == L'\r')) {
      --got;
    }
    *line = std::wstring(buffer, buffer + got);
    return true;
  }
  std::string bytes;
  if (!std::getline(std::cin, bytes)) {
    return false;
  }
  if (!bytes.empty() && bytes.back() == '\r') {
    bytes.pop_back();
  }
  *line = widen_utf8(bytes);
  return true;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  try {
    std::filesystem::path root;
    std::filesystem::path runtime;
    std::filesystem::path model;
    std::wstring prompt;
    bool prompt_given = false;
    int predict = kDefaultPredict;
    int context = kDefaultContext;
    int batch = kDefaultBatch;
    int ubatch = kDefaultUbatch;
    bool embedding_gpu = true;
    for (int i = 1; i < argc; ++i) {
      std::wstring arg = argv[i];
      if (arg == L"--help" || arg == L"-h") {
        usage();
        return 0;
      }
      if ((arg == L"--root" || arg == L"--runtime" || arg == L"--model" || arg == L"--prompt" ||
           arg == L"--predict" || arg == L"--ctx-size" || arg == L"--batch-size" || arg == L"--ubatch-size" ||
           arg == L"--embedding-gpu") &&
          i + 1 >= argc) {
        usage();
        return 2;
      }
      if (arg == L"--root") {
        root = argv[++i];
      } else if (arg == L"--runtime") {
        runtime = argv[++i];
      } else if (arg == L"--model") {
        model = argv[++i];
      } else if (arg == L"--prompt") {
        prompt = argv[++i];
        prompt_given = true;
      } else if (arg == L"--predict") {
        predict = std::stoi(std::wstring(argv[++i]));
        if (predict < 1) {
          throw std::runtime_error("--predict must be at least 1.");
        }
      } else if (arg == L"--ctx-size") {
        context = std::stoi(std::wstring(argv[++i]));
        if (context < 1) {
          throw std::runtime_error("--ctx-size must be at least 1.");
        }
      } else if (arg == L"--batch-size") {
        batch = std::stoi(std::wstring(argv[++i]));
      } else if (arg == L"--ubatch-size") {
        ubatch = std::stoi(std::wstring(argv[++i]));
      } else if (arg == L"--embedding-gpu") {
        std::wstring value = argv[++i];
        if (value == L"on") {
          embedding_gpu = true;
        } else if (value == L"off") {
          embedding_gpu = false;
        } else {
          throw std::runtime_error("--embedding-gpu expects on or off.");
        }
      } else {
        usage();
        return 2;
      }
    }
    if (root.empty() && (runtime.empty() || model.empty())) {
      root = find_root();
    }
    if (runtime.empty()) {
      if (root.empty()) {
        throw std::runtime_error("Pass --runtime or run from an ExVRAM checkout.");
      }
      runtime = root / kRuntimeRelative;
    }
    if (model.empty()) {
      if (root.empty()) {
        throw std::runtime_error("Pass --model or run from an ExVRAM checkout.");
      }
      model = root / kModelRelative;
    }
    if (!std::filesystem::exists(runtime) || !std::filesystem::exists(runtime.parent_path() / L"ggml-cuda.dll")) {
      throw std::runtime_error("The CUDA llama-server build is not next to ggml-cuda.dll.");
    }
    if (!std::filesystem::exists(model)) {
      throw std::runtime_error("Model file does not exist.");
    }

    CudaDevice device = query_cuda_device();
    INTERNET_PORT port = free_local_port();
    std::wcout << L"cuda_device: " << device.name << L"\n";
    std::wcout << L"compute_capability: " << device.major << L"." << device.minor << L"\n";
    std::wcout << L"multiprocessors: " << device.multiprocessors << L"\n";
    std::wcout << L"cuda_library: " << (runtime.parent_path() / L"ggml-cuda.dll").wstring() << L"\n";
    std::wcout << L"model: " << model.wstring() << L"\n";
    std::wcout << L"listen: 127.0.0.1:" << port << L"\n";

    GpuServer server;
    server.start(runtime, model, port, context, batch, ubatch, embedding_gpu);
    server.wait_until_ready();
    require_cuda(server, embedding_gpu);
    if (prompt_given) {
      std::cout << "answer: " << server.ask(prompt, predict) << "\n";
      server.stop();
      return 0;
    }
    std::wcout << L"ready. /exit stops the server.\n";
    for (;;) {
      std::wcout << L"> " << std::flush;
      std::wstring line;
      if (!read_prompt_line(&line) || line == L"/exit" || line == L"/quit") {
        break;
      }
      if (line.empty()) {
        continue;
      }
      std::cout << "answer: " << server.ask(line, predict) << "\n";
    }
    server.stop();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
