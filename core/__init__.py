"""
MAC-SQL Core
===========

This package implements the core components of the MAC-SQL framework:
- Agents (Selector, Decomposer, Refiner)
- Chat Manager for agent coordination
"""

from core.agents import SelectorAgent, DecomposerAgent, RefinerAgent
from core.chat_manager import ChatManager

__all__ = [
    'SelectorAgent',
    'DecomposerAgent',
    'RefinerAgent',
    'ChatManager'
] 