from typing import Literal

ARMOR_TYPE = Literal['none', 'light', 'medium', 'heavy']


class Armor:
    def __init__(self, name:str, ac:int, armor_type:ARMOR_TYPE, max_dex_bonus:int, dex_penalty = 0):
        self.name = name
        self.acBonus = ac
        self.dexPenalty = dex_penalty
        self.maxDexBonus = max_dex_bonus
        self.type = armor_type

class Unarmored(Armor):
    def __init__(self):
        super().__init__(
            name='Unarmored',
            ac=0,
            armor_type='none',
            max_dex_bonus=100)

class StuddedLeather(Armor):
    def __init__(self):
        super().__init__(
            name='Studded Leather Armor',
            ac=2,
            armor_type='light',
            max_dex_bonus=5,
            dex_penalty=-1)


class FullPlate(Armor):
    def __init__(self):
        super().__init__(
            name='Full Plate Armor',
            ac= 9,
            armor_type='heavy',
            max_dex_bonus= 1,
            dex_penalty= -6)

class Weapon:
    def __init__(self, name:str, ranged:bool, base_die:str, crit_range:int, crit_mul:int, damage_type:str,
                 two_handed:bool = False, reach:int = 1, finesse:bool = False):
        self.name = name
        self.ranged = ranged
        self.baseDie = base_die
        self.critRange = crit_range
        self.critMul = crit_mul
        self.damage_type = damage_type

        # Additional weapon traits
        self.twoHanded = two_handed
        self.reach = reach # Melee weapon trait
        self.finesse = finesse #Melee weapon trait


class UnarmedStrike(Weapon):
    def __init__(self, name: str = 'Unarmed Strike',
                 ranged: bool = False,
                 base_die: str = 'd4',
                 crit_range: int = 20,
                 crit_mul: int = 2,
                 damage_type: str = 'bludgeoning'):
        super().__init__(name, ranged, base_die, crit_range, crit_mul, damage_type)
class Longsword(Weapon):
    def __init__(self, name: str = 'Longsword',
                 ranged: bool = False,
                 base_die: str = 'd8',
                 crit_range: int = 19,
                 crit_mul: int = 2,
                 damage_type: str = 'slashing'):
        super().__init__(name, ranged, base_die, crit_range, crit_mul, damage_type)

class SmallScimitar(Weapon):
    def __init__(self, name: str = 'Small Scimitar',
                 ranged: bool = False,
                 base_die: str = 'd4',
                 crit_range: int = 18,
                 crit_mul: int = 2,
                 damage_type: str = 'slashing'):
        super().__init__(name, ranged, base_die, crit_range, crit_mul, damage_type)

class Greatsword(Weapon):
    def __init__(self, name: str = 'Greatsword',
                 ranged: bool = False,
                 base_die: str = 'd12',
                 crit_range: int = 19,
                 crit_mul: int = 2,
                 damage_type: str = 'slashing'):
        super().__init__(name, ranged, base_die, crit_range, crit_mul, damage_type)
        self.twoHanded = True

class Longbow(Weapon):
    def __init__(self, name: str = 'Longbow',
                 ranged: bool = True,
                 base_die: str = 'd8',
                 crit_range: int = 20,
                 crit_mul: int = 3,
                 damage_type: str = 'piercing'):
        super().__init__(name, ranged, base_die, crit_range, crit_mul, damage_type)
        self.twoHanded = True





