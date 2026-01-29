"""
Generator-based implementation for bio-constrained DNA codes using DP cache.

This approach uses dynamic programming to store only counts of valid sequences
rather than enumerating all codewords, achieving massive storage reduction.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.utils.evaluation import info_density

@dataclass(frozen=True)
class State:
    """
    State representation for DP cache.
    
    Attributes:
        last_sym: Last symbol (None if at start)
        run_len: Current run length of last symbol (0 if at start)
        gc_count: Number of G/C symbols seen so far
    """
    last_sym: Optional[str]
    run_len: int
    gc_count: int
    
    def __repr__(self) -> str:
        sym = self.last_sym or "START"
        return f"State({sym},run={self.run_len},gc={self.gc_count})"


class CRLLCodeGenerator:
    """
    Generator for bio-constrained DNA codes using DP cache.
    """
    
    def __init__(
        self,
        length: int = 12,
        max_run: int = 3,
        gc_lower: float = 0.4,
        gc_upper: float = 0.6,
        alphabet: Tuple[str, ...] = ('A', 'C', 'G', 'T'),
    ):
        """
        Initialize the CRLL code generator.
        
        Args:
            length: Codeword length to support
            max_run: Maximum homopolymer run (e.g., 3 means AAA allowed, AAAA not)
            gc_lower: Minimum GC content ratio
            gc_upper: Maximum GC content ratio
            alphabet: DNA alphabet (must be 4 symbols)
        """
        assert len(alphabet) == 4
        assert 0 <= gc_lower <= gc_upper <= 1
        assert max_run >= 1
        assert length >= 1
        
        self.length = length
        self.max_run = max_run
        self.gc_lower = gc_lower
        self.gc_upper = gc_upper
        self.alphabet = alphabet
        
        self.at_symbols = {'A', 'T'}
        self.gc_symbols = {'G', 'C'}

        self.min_gc = math.ceil(self.gc_lower * self.length)
        self.max_gc = math.floor(self.gc_upper * self.length)
        
        self.cache: Dict[Tuple[State, int], int] = {}
        
        print("building dp cache...")
        self._build_cache()
        print(f"dp cache built! size: {len(self.cache)} entries")
        print(f"estimated storage: {len(self.cache) * 8} bytes (~{len(self.cache) * 8 / 1024:.2f} KB)")
        print(f"I'm able to encode binary values of up to {math.floor(math.log2(self.get_capacity()))} bits!")
    
    def _get_valid_next_symbols(self, state: State, remaining: int) -> List[str]:
        """
        Get valid next symbols given current state.
        
        Checks:
        1. Homopolymer constraint (no more than max_run consecutive)
        2. GC balance constraint (gc_count can still reach bounds)
        """
        valid = []
        
        for sym in self.alphabet:
            if state.last_sym == sym and state.run_len >= self.max_run:
                continue
            
            next_gc = state.gc_count + (1 if sym in self.gc_symbols else 0)
            
            remaining_after = remaining - 1
            
            if next_gc > self.max_gc:
                continue
            if next_gc + remaining_after < self.min_gc:
                continue

            valid.append(sym)
        
        return valid
    
    def _transition(self, state: State, symbol: str) -> State:
        """
        Compute next state after adding a symbol.
        """
        new_gc = state.gc_count + (1 if symbol in self.gc_symbols else 0)
        if symbol == state.last_sym:
            new_run_len = state.run_len + 1
        else:
            new_run_len = 1
        
        return State(
            last_sym=symbol,
            run_len=new_run_len,
            gc_count=new_gc,
        )
    
    def _count_valid_sequences(self, state: State, remaining: int) -> int:
        """
        Count valid sequences from this state with given remaining length.
        
        Uses memoization (DP cache) for efficiency.
        """
        cache_key = (state, remaining)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if state.gc_count > self.max_gc:
            self.cache[cache_key] = 0
            return 0
        if state.gc_count + remaining < self.min_gc:
            self.cache[cache_key] = 0
            return 0

        if remaining == 0:
            result = 1 if self.min_gc <= state.gc_count <= self.max_gc else 0
            self.cache[cache_key] = result
            return result

        valid_syms = self._get_valid_next_symbols(state, remaining)
        total = 0
        for sym in valid_syms:
            next_state = self._transition(state, sym)
            total += self._count_valid_sequences(next_state, remaining - 1)
        
        self.cache[cache_key] = total
        return total
    
    def _build_cache(self) -> None:
        """
        Pre-compute the DP cache for all states and lengths.
        """
        states = []

        start_state = State(None, 0, 0)
        states.append(start_state)

        for sym in self.alphabet:
            for run_len in range(1, self.max_run + 1):
                for gc_count in range(0, self.length + 1):
                    state = State(sym, run_len, gc_count)
                    states.append(state)

        for state in states:
            for length in range(0, self.length + 1):
                self._count_valid_sequences(state, length)
    
    def encode(self, binary_value: int) -> str:
        """
        Encode a binary value into a DNA sequence of the configured length.
        
        Args:
            binary_value: Non-negative integer to encode
        Returns:
            DNA sequence as string
            
        Raises:
            ValueError: If binary_value is too large for given length
        """
        initial_state = State(None, 0, 0)
        total_capacity = self._count_valid_sequences(initial_state, self.length)
        
        if binary_value >= total_capacity:
            raise ValueError(
                f"Binary value {binary_value} too large for length {self.length} "
                f"(capacity: {total_capacity})"
            )
        
        sequence = []
        state = initial_state
        remaining_value = binary_value
        
        for pos in range(self.length):
            remaining_length = self.length - pos
            valid_syms = self._get_valid_next_symbols(state, remaining_length)
            
            counts = []
            for sym in valid_syms:
                next_state = self._transition(state, sym)
                count = self._count_valid_sequences(next_state, remaining_length - 1)
                counts.append(count)
            
            cumsum = 0
            selected_sym = None
            for sym, count in zip(valid_syms, counts):
                if remaining_value < cumsum + count:
                    selected_sym = sym
                    remaining_value -= cumsum
                    break
                cumsum += count
            
            if selected_sym is None:
                raise RuntimeError("Failed to select symbol (should never happen)")
            
            sequence.append(selected_sym)
            state = self._transition(state, selected_sym)
        
        return ''.join(sequence)
    
    def decode(self, dna_sequence: str) -> int:
        """
        Decode a DNA sequence back to its binary value.
        
        Args:
            dna_sequence: DNA sequence string
            
        Returns:
            Binary value (non-negative integer)
        """
        length = len(dna_sequence)
        if length != self.length:
            raise ValueError(f"Sequence length {length} does not match length {self.length}")
        
        state = State(None, 0, 0)
        binary_value = 0
        
        for pos, sym in enumerate(dna_sequence):
            remaining_length = length - pos
            valid_syms = self._get_valid_next_symbols(state, remaining_length)
            
            counts = []
            for valid_sym in valid_syms:
                next_state = self._transition(state, valid_sym)
                count = self._count_valid_sequences(next_state, remaining_length - 1)
                counts.append(count)
            
            cumsum = 0
            for valid_sym, count in zip(valid_syms, counts):
                if valid_sym == sym:
                    binary_value += cumsum
                    break
                cumsum += count
            else:
                raise ValueError(f"Invalid symbol '{sym}' at position {pos}")
            
            state = self._transition(state, sym)
        
        return binary_value
    
    def get_capacity(self) -> int:
        """
        Get the number of valid sequences of the configured length.
        
        Returns:
            Number of valid DNA sequences
        """
        initial_state = State(None, 0, 0)
        return self._count_valid_sequences(initial_state, self.length)
    
    def information_density(self) -> float:
        """
        Get the information density for the configured length.
        
        Returns:
            Information density in bits/nt
        """
        capacity = self.get_capacity()
        if capacity == 0:
            return 0.0
        return info_density(capacity, self.length)

if __name__ == "__main__":
    gen = CRLLCodeGenerator(length=10, max_run=3, gc_lower=0.4, gc_upper=0.6)
    print("information density: ", gen.information_density())

