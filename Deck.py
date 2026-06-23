# French card implementation for RL game

from random import shuffle
import numpy as np
from bitarray import bitarray

class Deck:
    """
    A class representing a deck of playing cards.
    """
    def __init__(self, num_cards=52, suits=4, jacks=2):
        """
        The default number of cards is 52, but it can be customized by providing the num_cards parameter.
        The suit_order is a list that defines the order of suits in the deck.
        To the total number of cards, we add 2 jokers.
        """
        self.num_cards = num_cards - jacks # Add 2 jokers to the total number of cards
        self.suits = suits
        self.cards = [i for i in range(self.num_cards)]
        self.jacks = jacks
        shuffle(self.cards)

    def draw_card(self):
        """
        Draw a card from the deck. If the deck is empty, it will return None.
        """
        if len(self.cards) == 0:
            return None
        return self.cards.pop()
    
    def draw_cards(self, num: int) -> list[int]:
        """
        Draw multiple cards from the deck. If the deck has fewer cards than requested, it will return all remaining cards.
        """
        drawn_cards = []
        for _ in range(num):
            card = self.draw_card()
            if card is not None:
                drawn_cards.append(card)
            else:
                break
        return drawn_cards
    
    def reset_deck(self):
        """
        Reset the deck to its original state and shuffle it.
        """
        self.cards = [i for i in range(self.num_cards)]
        shuffle(self.cards)

class DeckHelper:
    """
    A helper class for the Deck class to provide additional functionalities.
    """
    @staticmethod
    def is_joker(deck, card):
        """
        Check if a card is a joker based on its number.
        """
        return card >= deck.num_cards

    """
    A helper class for the Deck class to provide additional functionalities.
    """
    @staticmethod
    def card_to_suit(deck, card):
        """
        Convert a card number to its corresponding suit based on the suit order.
        If joker, it will return -1 as the suit.
        """
        cards_per_suit = (deck.num_cards) // deck.suits  # Exclude jokers
        if DeckHelper.is_joker(deck, card):
            return -1  # Joker has no suit
        return card // cards_per_suit

    @staticmethod
    def card_to_rank(deck, card) -> int:
        """
        Convert a card number to its corresponding rank based on the total number of cards in the deck.
        If joker, it will return 0 as the rank.
        """
        cards_per_suit = (deck.num_cards) // deck.suits  # Exclude jokers
        if DeckHelper.is_joker(deck, card):
            return 0
        return (card % cards_per_suit) + 1
    
    @staticmethod
    def card_to_string(deck, card):
        """
        Convert a card number to a human-readable string format (e.g., "Ace of Hearts").
        """
        rank = DeckHelper.card_to_rank(deck, card)
        suit = DeckHelper.card_to_suit(deck, card)
        rank_names = {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"}
        rank_str = rank_names.get(rank, str(rank))
        return f"{rank_str} of {suit}"  
    
    @staticmethod
    def card_to_bitarray(num_cards, card):
        """
        Convert a card number to a one-hot encoded vector representation.
        The length of the vector is equal to the total number of cards in the deck.
        """
        b = bitarray(num_cards)
        b.setall(0)
        b[card] = 1
        return b

    @staticmethod
    def cards_to_bitarray(num_cards: int, cards: list[int]):
        """
        Convert a list of card numbers to a combined one-hot encoded vector representation.
        The length of the vector is equal to the total number of cards in the deck.
        """
        combined_vector = bitarray(num_cards)
        combined_vector.setall(0)
        for card in cards:
            combined_vector[card] = 1
        return combined_vector
    
    @staticmethod
    def bitarray_to_cards(num_cards, bitarray_cards):
        """
        Convert a one-hot encoded vector representation back to a list of card numbers.
        The length of the input vector should be equal to the total number of cards in the deck.
        """
        cards = []
        for i in range(num_cards):
            if bitarray_cards[i]:
                cards.append(i + 1)  # Card numbers are 1-indexed
        return cards

    @staticmethod
    def add_cards_to_vector(vector_A, vector_B):
        """
        Adds the cards represented by the list of card numbers to the existing vector,
        effectively updating the hand represented by the vector to include those cards.
        """
        return vector_A | vector_B
    
    @staticmethod
    def remove_cards_from_vector(vector_A, vector_B):
        """
        Subtracts the cards represented by vector_B from vector_A, 
        effectively removing those cards from the hand represented by vector_A.
        """
        return vector_A & ~vector_B
    
    @staticmethod
    def aggregate_bitarrays(bitarrays: list[bitarray]) -> bitarray:
        """
        Aggregates a list of bitarrays into a single bitarray by performing a bitwise OR operation.
        This is useful for combining the cards played by multiple players in a round.
        """
        aggregated = bitarray(len(bitarrays[0]))
        aggregated.setall(0)
        for b in bitarrays:
            try:
                aggregated |= b
            except Exception as e:
                print(f"Error aggregating bitarrays: \t{e}")
                print(f"Current aggregated: \t{aggregated}")
                print(f"length of aggregated: \t{len(aggregated)}")
                print(f"Current bitarray: \t{b}")
                print(f"length of current bitarray: \t{len(b)}")
                raise e
        return aggregated
    

    @staticmethod
    def bitarray_to_bytes_array(bits: bitarray, num_bits: int = 53) -> np.ndarray:
        """
        Converte un bitarray di num_bits bit in un array numpy di uint8.
        
        Parametri:
            bits: bitarray di lunghezza num_bits
            num_bits: numero di bit da convertire (default 53)
        
        Restituisce:
            np.ndarray di dtype uint8, shape = (ceil(num_bits/8),)
        """
        # 1. Crea una copia per non modificare l'originale
        bits_copy = bits.copy()
        
        # 2. Padding a multiplo di 8 bit (aggiungi zeri in fondo)
        remainder = len(bits_copy) % 8
        if remainder != 0:
            bits_copy.extend([0] * (8 - remainder))
        
        # 3. Converti in bytes (bytes object)
        byte_data = bits_copy.tobytes()
        
        # 4. Converti in numpy array uint8
        return np.frombuffer(byte_data, dtype=np.uint8)
    

    @staticmethod
    def int_to_bitarray(num_cards: int, value: int) -> bitarray:
        """
        Convert an integer to a bitarray of length num_cards.
        The least significant bit corresponds to the first card.
        """
        b = bitarray(num_cards)
        b.setall(0)
        for i in range(num_cards):
            if (value >> i) & 1:
                b[i] = 1
        return b