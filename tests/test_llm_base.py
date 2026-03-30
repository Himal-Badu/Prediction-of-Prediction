import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import torch
import pytest


class TestLLMImports:
    def test_import(self):
        from pop.core.llm_base import LLMBase, create_llm
        assert LLMBase is not None
        assert create_llm is not None
