from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 禁用代理，防止干扰 DeepSeek 连接
os.environ['NO_PROXY'] = 'api.deepseek.com'

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def _load_default_env() -> None:
    env_paths = [Path(".env"), Path(__file__).resolve().parents[1] / ".env"]
    if load_dotenv is not None:
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
        return
    for env_path in env_paths:
        if not env_path.exists(): continue
        with env_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

_load_default_env()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

@dataclass
class LLMConfig:
    backend: str
    openai: Dict[str, Any]
    ollama: Dict[str, Any]
    openrouter: Dict[str, Any]
    deepseek: Dict[str, Any]
    siliconflow: Dict[str, Any]
    custom: Dict[str, Any]

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.backend = config.backend.lower()
        self._openai_client: Optional[OpenAI] = None
        
        # 初始化 Session
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LLMClient":
        return cls(
            LLMConfig(
                backend=config.get("backend", "openai"),
                openai=config.get("openai", {}),
                ollama=config.get("ollama", {}),
                openrouter=config.get("openrouter", {}),
                deepseek=config.get("deepseek", {}),
                siliconflow=config.get("siliconflow", {}),
                custom=config.get("custom", {}),
            )
        )

    @property
    def openai_client(self) -> OpenAI:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def generate(self, prompt: str, *, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        backend_map = {
            "openai": self._generate_openai,
            "ollama": self._generate_ollama,
            "openrouter": self._generate_openrouter,
            "deepseek": self._generate_deepseek,
            "siliconflow": self._generate_siliconflow,
            "custom": self._generate_custom,
        }
        handler = backend_map.get(self.backend)
        if not handler:
            raise ValueError(f"Unsupported LLM backend: {self.backend}")
        return handler(prompt, system_prompt=system_prompt, **kwargs)

    def _generate_openai_compatible(
        self, prompt: str, system_prompt: Optional[str], cfg: Dict[str, Any], default_url: str, env_key: str, **kwargs: Any
    ) -> str:
        # --- 1. 必须首先定义 api_key ---
        current_api_key = cfg.get("api_key") or os.getenv(env_key)
        if not current_api_key:
            raise RuntimeError(f"API key for {env_key} not found.")

        # --- 2. 构造参数 ---
        model = cfg.get("model")
        temperature = kwargs.get("temperature", cfg.get("temperature", 0.0))
        max_tokens = kwargs.get("max_tokens", cfg.get("max_tokens"))
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # --- 3. 构造 Headers ---
        url = cfg.get("url", default_url)
        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json",
            "Connection": "close"
        }
        if cfg.get("headers"):
            headers.update(cfg["headers"])

        # --- 4. 执行请求 ---
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=cfg.get("timeout", 120))
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ChunkedEncodingError:
            time.sleep(2)
            response = self.session.post(url, json=payload, headers=headers, timeout=120)
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"LLM 请求失败 ({self.backend}): {e}")

    def _generate_deepseek(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        cfg = self.config.deepseek
        if "model" not in cfg: cfg["model"] = "deepseek-chat"
        return self._generate_openai_compatible(
            prompt, system_prompt, cfg, 
            default_url="https://api.deepseek.com/chat/completions",
            env_key="DEEPSEEK_API_KEY", **kwargs
        )

    def _generate_siliconflow(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        return self._generate_openai_compatible(
            prompt, system_prompt, self.config.siliconflow, 
            default_url="https://api.siliconflow.cn/v1/chat/completions",
            env_key="SILICONFLOW_API_KEY", **kwargs
        )

    def _generate_custom(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        return self._generate_openai_compatible(
            prompt, system_prompt, self.config.custom, 
            default_url="", env_key="CUSTOM_LLM_API_KEY", **kwargs
        )

    def _generate_openai(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        model = self.config.openai.get("model", "gpt-4o-mini")
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.openai_client.chat.completions.create(
            model=model, messages=messages, 
            temperature=kwargs.get("temperature", self.config.openai.get("temperature", 0.0))
        )
        return response.choices[0].message.content.strip()

    def _generate_ollama(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        payload = {
            "model": self.config.ollama.get("model", "mistral"),
            "prompt": prompt, "system": system_prompt or "", "stream": False,
            "options": self.config.ollama.get("options", {})
        }
        url = self.config.ollama.get("url", "http://localhost:11434/api/generate")
        response = requests.post(url, json=payload, timeout=self.config.ollama.get("timeout", 120))
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _generate_openrouter(self, prompt: str, *, system_prompt: Optional[str], **kwargs: Any) -> str:
        return self._generate_openai_compatible(
            prompt, system_prompt, self.config.openrouter,
            default_url="https://openrouter.ai/api/v1/chat/completions",
            env_key="OPENROUTER_API_KEY", **kwargs
        )