from typing import Literal

ARMOR_TYPE = Literal['light', 'medium', 'heavy']


class Armor:
    def __init__(self, name:str, ac:int, armor_type:ARMOR_TYPE, max_dex_bonus:int, dex_penalty = 0):
        self.name = name
        self.acBonus = ac
        self.dexPenalty = dex_penalty
        self.maxDexBonus = max_dex_bonus
        self.type = armor_type

    def getAcBonus(self):
        return self.acBonus

class StuddedLeather(Armor):
    def __init__(self, name: str = 'Studded Leather Armor',
                 ac: int = 3,
                 armor_type: ARMOR_TYPE = 'light',
                 max_dex_bonus: int = 5):
        super().__init__(name, ac, armor_type, max_dex_bonus)
        self.dexPenalty = -1

class FullPlate(Armor):
    def __init__(self, name: str = 'Full Plate Armor',
                 ac: int = 9,
                 armor_type: ARMOR_TYPE = 'heavy',
                 max_dex_bonus: int = 1):
        super().__init__(name, ac, armor_type, max_dex_bonus)
        self.dexPenalty = -6

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


class Longsword(Weapon):
    def __init__(self, name: str = 'Longsword',
                 ranged: bool = False,
                 base_die: str = 'd8',
                 crit_range: int = 19,
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





