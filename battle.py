#!/usr/bin/env python

from __future__ import annotations # allows forward declaration of types
import csv
import enum
import os
import random
import re
import time
from typing import Literal, Union

import numpy as np
import torch
from torch import multiprocessing
from tqdm import trange

import ppo_clip
rnd = np.random.default_rng()

# Import armor and weapons
from arsenal import *

# Setup vars
MAX_TURNS = 100
MAX_CHARS = 4 + 1
OBS_SIZE = 3 * MAX_CHARS
HP_SCALE = 75 # roughly, max HP across all entities in the battle (but a fixed constant, not rolled dice!)

Ability = enum.IntEnum('Ability', 'STR DEX CON INT WIS CHA', start=0)
Saving_Throw = enum.IntEnum('Saving Throw', 'REF FORT WILL', start=0)
Experiment_Name = 'Medusa_Nostrat'

epoch_id = encounter_id = round_id = -1
actions_csv = csv.writer(open(f"release/{Experiment_Name}/actions_{os.getpid()}.csv", "w"))
actions_csv.writerow('epoch encounter round actor action target t_fullDefense t_weakest raw_hp obs_hp'.split())
outcomes_csv = csv.writer(open(f"release/{Experiment_Name}/outcomes_{os.getpid()}.csv", "w"))
outcomes_csv.writerow('epoch encounter num_rounds actor team team_win max_hp final_hp'.split())

class Dice:
    def __init__(self, XdY: str):
        m = re.search(r'([1-9][0-9]*)?d([1-9][0-9]*)([+-][1-9][0-9]*)?', XdY)
        g = m.groups()
        self.num_dice = int(g[0] or 1)
        self.dice_type = int(g[1])
        self.bonus = int(g[2] or 0)
    def roll(self, crit_hit: bool = False, crit_mul: int = 2) -> int:
        rolls = rnd.integers(low=1, high=self.dice_type, endpoint=True, size=self.num_dice * (crit_mul if crit_hit else 1))
        # D&D rules: even if bonus is negative, total can't fall below 1. Kept for pathfinder simplicity.
        result = max(1, rolls.sum() + self.bonus * (crit_mul if crit_hit else 1))
        #print(rolls, result)
        return result

class D20:
    def __init__(self, bonus: int):
        self.bonus = bonus
    def roll(self):
        base_roll = rnd.integers(low=1, high=20, endpoint=True)
        mod_roll = base_roll + self.bonus
        return mod_roll, base_roll

def roll(XdY: str):
    return Dice(XdY).roll()

class Character:
    def __init__(self, name: str, team: int, hp: int, bab: int, armor: Armor,
                 meleeWeapon: Weapon, rangedWeapon: Weapon, actions: list[Action],
                 ability_mods: list[int] = [0]*6, saving_throws: list[int] = None,
                 spells: list[int] = [0]*9, spell_save=10,
                 initiative_boost=0, melee_boost=0, ranged_boost=0, ranged_damage_boost=0):
        self.name = name
        self.team = team

        # Base Stats
        self.max_hp = self.hp = hp
        self.ability_mods = list(ability_mods)
        self.saving_throws = list(saving_throws) # if None, default == to ability_mods
        self.death_treshold = -10 - self.ability_mods[Ability.CON] * 2
        self.armor = armor
        self.meleeWeapon = meleeWeapon
        self.rangedWeapon = rangedWeapon
        self.bab = bab
        self.actions = actions

        self.initiativeBuff = initiative_boost
        self.meleeBuff = melee_boost
        self.rangedBuff = ranged_boost
        self.rangedDamageBuff = ranged_damage_boost
        self.infer_stats()

        # Spellcasting
        self.max_spells = np.array(spells, dtype=int)
        self.curr_spells = self.max_spells.copy()
        self.spell_save_dc = spell_save

        # Status
        self.unconscious = False
        self.petrified = False
        self.dead = False
        self.coma = False
        self.prone = False
        self.protected = False

        # Medusa fighting strategy
        self.accuracy = 1
        self.avoidGaze = 0

        # State update
        self.start_of_round() # initializes some properties
        self.flatFooted = True # Everyone starts flat-footed

    def infer_stats(self):
        # Infered stats
        self.ac = 10 + min(self.ability_mods[Ability.DEX], self.armor.maxDexBonus) + self.armor.acBonus
        self.touchAc = 10 + min(self.ability_mods[Ability.DEX], self.armor.maxDexBonus)
        self.flatFootedAc = 10 + self.armor.acBonus
        self.cmd = 10 + min(self.ability_mods[Ability.DEX], self.armor.maxDexBonus) + self.ability_mods[
            Ability.STR] + self.bab
        self.cmb = min(self.ability_mods[Ability.DEX], self.armor.maxDexBonus) + self.ability_mods[
            Ability.STR] + self.bab

        self.initiativeBonus = self.ability_mods[Ability.DEX] + self.initiativeBuff

        if self.meleeWeapon.finesse:
            self.meleeToHit = self.bab + self.ability_mods[Ability.DEX] + self.meleeBuff
        else:
            self.meleeToHit = self.bab + self.ability_mods[Ability.STR] + self.meleeBuff

        if self.meleeWeapon.twoHanded:
            self.meleeDamageDie = Dice(
                str(1) + self.meleeWeapon.baseDie + '+' + str(int(self.ability_mods[Ability.STR]) * 1.5))
            self.meleeDamagePADie = Dice(
                str(1) + self.meleeWeapon.baseDie + '+' + str(int((self.ability_mods[Ability.STR]) * 1.5) + 3))
        else:
            self.meleeDamageDie = Dice(str(1) + self.meleeWeapon.baseDie + '+' + str(self.ability_mods[Ability.STR]))
            self.meleeDamagePADie = Dice(
                str(1) + self.meleeWeapon.baseDie + '+' + str(self.ability_mods[Ability.STR] + 2))

        self.rangedToHit = self.bab + self.ability_mods[Ability.DEX] + self.rangedBuff
        if self.rangedWeapon:
            self.rangedDamageDie = Dice(str(1) + self.rangedWeapon.baseDie + '+' + str(self.rangedDamageBuff))
            self.rangedDSDamageDie = Dice(str(1) + self.rangedWeapon.baseDie + '+' + str(self.rangedDamageBuff + 2))

    def get_ac(self, ff):
        ac = self.ac
        if self.fullDefense:
            ac += 4
        if self.flatFooted or ff or self.petrified:
            ac = self.flatFootedAc
        if self.prone:
            ac -= 4
        if self.protected:
            ac += 2
        return ac

    def get_touch_ac(self, ff):
        ac = self.touchAc
        if self.fullDefense:
            ac += 4
        if self.flatFooted or ff:
            ac = 10
        if self.prone:
            ac -= 4
        if self.protected:
            ac += 2
        return ac

    def start_of_round(self):
        self.fullDefense = False
        self.flatFooted = False
        self.prone = False

    def end_of_encounter(self, env):
        pass

    def saving_throw(self, save_dc: int, saving_throw: Saving_Throw):
        "Returns True if the save succeeds and False if it fails"
        if self.coma and saving_throw in [Saving_Throw.REF, Saving_Throw.FORT]:
            return False
        mod_roll, nat_roll = D20(self.saving_throws[saving_throw] + 2*self.protected).roll()
        if nat_roll == 1: return False
        elif nat_roll == 20: return True
        else: return mod_roll >= save_dc

    def damage(self, dmg_hp: int, dmg_type: str):
        "Apply damage to this character"
        self.hp = max(self.death_treshold, self.hp - dmg_hp)
        if self.hp < 0:
            self.coma = True
            if self.hp == self.death_treshold:
                self.dead = True
        self.unconscious = False

    def heal(self, heal_hp: int):
        "Apply healing to this character"
        if self.hp <= self.death_treshold:
            pass
        else:
            self.hp = min(self.max_hp, self.hp + heal_hp)
            if self.hp >= 0:
                self.coma = False

    def takeStrDamage(self, damage:int):
        self.ability_mods[Ability.STR] -= damage
        if self.ability_mods[Ability.STR] <= -5:
            self.coma = True
        self.infer_stats()

class RandomCharacter(Character):
    def act(self, env):
        actions = [a for a in self.actions if not a.is_forbidden(self, env)]
        if not actions:
            #print(f"{self.name} has no allowable actions")
            return
        action = rnd.choice(actions)
        targets = [c for c in env.characters if action.plausible_target(self, c)]
        if not targets:
            #print(f"{self.name} could not find a target for {action.name}")
            return
        target = rnd.choice(targets)
        action(actor=self, target=target, env=env)


class PPOStrategy:
    def __init__(self, n_acts):
        self.n_acts = n_acts * MAX_CHARS
        self.obs_dim = OBS_SIZE
        self.act_crit = ppo_clip.MLPActorCritic(self.obs_dim, self.n_acts, hidden_sizes=[32])
        self.act_crit.share_memory() # docs say this is required, but doesn't seem to be?
        # https://bair.berkeley.edu/blog/2021/07/14/mappo/ suggests that smaller clip (0.2) and
        # fewer iters (5-15) stabilizes learning with PPO in multi-agent settings?
        # So far, I don't see a benefit.
        self.optim = ppo_clip.PPOAlgo(self.act_crit) # pi_lr=3e-4, vf_lr=1e-3, clip_ratio=0.2, train_pi_iters=15, train_v_iters=15
        self.encounters = 0

    def alloc_buf(self):
        # Wait to allocate the buffers until we're in worker processes, so we don't trample the same memory
        self.buf = ppo_clip.PPOBuffer(self.obs_dim, self.n_acts, act_dim=None, size=1000 * OBS_SIZE * MAX_TURNS)

    def end_of_encounter(self):
        self.encounters += 1

    def update(self, data):
        self.optim.update(data)

class PPOCharacter(Character):
    def __init__(self, ppo_strat, survival=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert ppo_strat.n_acts == len(self.actions) * MAX_CHARS
        self.ppo_strat = ppo_strat
        self.act_crit = ppo_strat.act_crit
        self.buf = ppo_strat.buf
        self.survival = survival
        self.old_hp_score = 0
        self.prev_OFAVL = None # state tuple from previous reward

    def end_of_encounter(self, env):
        # all consequences of last action are now apparent, so can calc reward for it
        self.save_experience(env)
        self.buf.finish_path(self.get_reward(env))
        self.ppo_strat.end_of_encounter()

    def get_obs(self, env):
        obs = []
        for c in env.characters:
            obs.extend([
                #c.team == self.team,           # On our team? Same for whole training run, so useless.
                #(c.ac - 10) / 10,              # Armor class.  Right now, does not change.
                c == self,                      # Ourself? maybe useful when 1 AI plays many monsters
                (c.max_hp - c.hp) / HP_SCALE,   # Absolute hp lost -- we can track this as a player.
                c.dead,                         # Death (HP below death treshold)
                #c.prone,                        # Pathfinder prone condition is visible
                # Below this point is cheating -- info not available to players, only DM
                #c.hp / HP_SCALE,               # current absolute health
                #c.max_hp / self.max_hp,        # Stronger or weaker than us? Varies if hp are rolled.
            ])
        return torch.tensor(obs, dtype=torch.float32)

    def get_hp_reward(self, env):
        """
        Reward characters when the enemies lose hp, or their team gains hp.
        Enemies losing 100% of their hp is worth +1,
        team losing 100% of their hp is worth -1.
        Each teammate death is worth an additional -1/team_size.
        At the start of the battle, with no dead and no hp losses, score should be zero.
        """
        chars = env.characters
        hp = np.empty(len(chars), dtype=float)
        max_hp = np.empty(len(chars), dtype=float)
        team = np.empty(len(chars), dtype=bool)
        for i, char in enumerate(chars):
            hp[i] = char.hp
            max_hp[i] = char.max_hp
            team[i] = (char.team == self.team)
        # All hp losses are equal, whether from a weak character or a strong one
        team_frac = hp[team].sum() / max_hp[team].sum()
        opp_frac = hp[~team].sum() / max_hp[~team].sum()
        team_size = team.sum()
        team_deaths = len([char for char in chars if char.dead])
        # If `survival` is 0, there's no special attempt to avoid deaths.
        # If `survival` is 1, it's better to have 2 characters at 1 hp than one dead and one full.
        # The default of 0.5 means avoiding death is worth half of a teammate's hp.
        # Positive team is ahead, negative team is behind, zero is balanced loss of hp
        hp_score = team_frac - opp_frac - self.survival*team_deaths/team_size
        # Positive team has gained ground, negative team has lost ground, zero is no or balanced changes
        hp_delta = hp_score - self.old_hp_score
        self.old_hp_score = hp_score
        return hp_delta

    def get_reward(self, env):
        return self.get_hp_reward(env)

    def save_experience(self, env):
        rew = self.get_reward(env)
        if self.prev_OFAVL is None:
            # First action.  Any HP losses before this are independent of our actions,
            # so shouldn't count toward our rewards.
            pass
        else:
            obs, fbn, act, val, logp = self.prev_OFAVL
            self.buf.store(obs, fbn, act, rew, val, logp)
        self.prev_OFAVL = None

    def act(self, env):
        # all consequences of last action are now apparent, so can calc reward for it
        self.save_experience(env)

        chars_acts = []
        fbn = []
        for c in env.characters:
            for a in self.actions:
                chars_acts.append((c,a))
                fbn.append(a.is_forbidden(self, env) or not a.plausible_target(self, c))
        fbn = np.array(fbn)
        if fbn.all():
            #print(f"{self.name} has no allowable action/target pairs")
            return

        with torch.no_grad():
            obs = self.get_obs(env)
            act, val, logp, pi = self.act_crit.step(obs, fbn)
            act_idx = act.item()
            target, action = chars_acts[act_idx]
        self.prev_OFAVL = (obs, fbn, act, val, logp)
        action(actor=self, target=target, env=env)

class Action:
    def is_forbidden(self, actor: Character, env: Environment):
        return False

    def plausible_target(self, actor: Character, target: Character):
        return True

    def _self_only(self, actor: Character, target: Character):
        # This is a good choice for Actions without an explicit target, like Dodge, so there's just one unique choice
        return actor == target

    def _conscious_ally(self, actor: Character, target: Character):
        return target.team == actor.team and not target.coma

    def _unprotected_ally(self, actor: Character, target: Character):
        return target.team == actor.team and not target.coma and not target.protected

    def _living_ally(self, actor: Character, target: Character):
        return target.team == actor.team and not target.dead

    def _unconscious_ally(self, actor: Character, target: Character):
        return target.team == actor.team and target.unconscious

    def _conscious_enemy(self, actor: Character, target: Character):
        return target.team != actor.team and not target.coma

class FullDefense(Action):
    name = 'Full Defense'
    plausible_target = Action._self_only

    def __call__(self, actor: Character, target: Character, env: Environment):
        actor.fullDefense = True
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, False, False, 0, 0])

class Awaken(Action):
    name = 'Awaken'
    plausible_target = Action._unconscious_ally

    def __call__(self, actor: Character, target: Character, env: Environment):
        target.unconscious = False
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, False, False, 0, 0])

class Trip(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead and not target.prone

    def __call__(self, actor: Character, target: Character, env: Environment):
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        # all([]) == True, which seems ok
        attack_roll, nat_roll = D20(actor.cmb).roll()
        if (attack_roll >= target.cmd or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) <= actor.accuracy:
            target.prone = True
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Trip',
                                  target.name, target.fullDefense, t_weakest, 0, 0])


class MeleeAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        ac_to_hit = target.get_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        # all([]) == True, which seems ok
        attack_roll, nat_roll = D20(actor.meleeToHit).roll()
        if (attack_roll >= ac_to_hit or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) <= actor.accuracy:
            crit_hit = (nat_roll >= actor.meleeWeapon.critRange) or target.unconscious or target.coma
            dmg_roll = actor.meleeDamageDie.roll(crit_hit=crit_hit, crit_mul=actor.meleeWeapon.critMul)
            before_hp = target.hp
            target.damage(dmg_roll, actor.meleeWeapon.damage_type)
            after_hp = target.hp
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.meleeWeapon.name,
                                  target.name, target.fullDefense, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.meleeWeapon.name,
                                  target.name, target.fullDefense, t_weakest, 0, 0])

class MeleePowerAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        ac_to_hit = target.get_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        # all([]) == True, which seems ok
        attack_roll, nat_roll = D20(actor.meleeToHit).roll()
        if (attack_roll -1 >= ac_to_hit or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) <= actor.accuracy:
            crit_hit = (nat_roll >= actor.meleeWeapon.critRange) or target.unconscious or target.coma
            dmg_roll = actor.meleeDamagePADie.roll(crit_hit=crit_hit, crit_mul=actor.meleeWeapon.critMul)
            before_hp = target.hp
            target.damage(dmg_roll, actor.meleeWeapon.damage_type)
            after_hp = target.hp
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.meleeWeapon.name+' Power Attack',
                                  target.name, target.fullDefense, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.meleeWeapon.name+' Power Attack',
                                  target.name, target.fullDefense, t_weakest, 0, 0])


class SpiderBite(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        ac_to_hit = target.get_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        # all([]) == True, which seems ok
        attack_roll, nat_roll = D20(actor.meleeToHit).roll()
        if (attack_roll >= ac_to_hit or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) <= actor.accuracy:
            crit_hit = (nat_roll >= actor.meleeWeapon.critRange) or target.unconscious or target.coma
            dmg_roll = actor.meleeDamageDie.roll(crit_hit=crit_hit, crit_mul=actor.meleeWeapon.critMul)
            before_hp = target.hp
            target.damage(dmg_roll, actor.meleeWeapon.damage_type)
            after_hp = target.hp

            if target.saving_throw(16, Saving_Throw.FORT) and random.uniform(0, 1) <= 0.5:
                target.takeStrDamage(1)

            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Spider Bite',
                                  target.name, target.fullDefense, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Spider Bite',
                                  target.name, target.fullDefense, t_weakest, 0, 0])

class UniqueMedusaFullDefense(Action):
    name = 'Full Defense'
    plausible_target = Action._self_only

    def __call__(self, actor: Character, target: Character, env: Environment):
        # Tries to petrify everyone
        for other in env.characters:
            if target.team != actor.team and not target.dead and random.uniform(0, 1) > actor.avoidGaze:
                other.dead = other.saving_throw(16, Saving_Throw.FORT)
                if other.dead:
                    before_hp = target.hp
                    other.hp = other.death_treshold
                    other.coma = True
                    t_weakest = all(target.hp <= other.hp for other in env.characters
                                    if self.plausible_target(actor, other) and not other.coma)
                    actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Petrification',
                                          target.name, target.fullDefense, t_weakest, -before_hp + other.hp,
                                          other.hp - before_hp])
                    return

        actor.fullDefense = True
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, False, False, 0, 0])

class UniqueMedusaMeleeAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        # Tries to petrify everyone
        for other in env.characters:
            if self.plausible_target(actor, other) and random.uniform(0, 1) > actor.avoidGaze:
                other.dead = other.saving_throw(16, Saving_Throw.FORT)
                if other.dead:
                    before_hp = target.hp
                    other.hp = other.death_treshold
                    other.coma = True
                    t_weakest = all(target.hp <= other.hp for other in env.characters
                                    if self.plausible_target(actor, other) and not other.coma)
                    actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Petrification',
                                          target.name, target.fullDefense, t_weakest, -before_hp + other.hp,
                                          other.hp - before_hp])
                    return

        ac_to_hit = target.get_ac(True)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        totalDamage = 0

        for bonus, dice, critrange in zip([10, 5, 5],[Dice('1d4'), Dice('1d4'), Dice('1d4')], [19, 19, 20]):
            attack_roll, nat_roll = D20(bonus).roll()
            if (attack_roll >= ac_to_hit or nat_roll == 20) and nat_roll != 1:
                crit_hit = (nat_roll >= critrange) or target.unconscious or target.coma
                dmg_roll = dice.roll(crit_hit=crit_hit, crit_mul=2)
                totalDamage += dmg_roll

        if totalDamage != 0:
            before_hp = target.hp
            target.damage(totalDamage, actor.meleeWeapon.damage_type)
            after_hp = target.hp
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Medusa Melee Attack',
                                  target.name, target.fullDefense, t_weakest, -totalDamage, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Medusa Melee Attack',
                                  target.name, target.fullDefense, t_weakest, 0, 0])


class UniqueMedusaRangedAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):

        # Tries to petrify everyone
        for other in env.characters:
            if self.plausible_target(actor, other) and random.uniform(0, 1) > actor.avoidGaze:
                other.dead = other.saving_throw(16, Saving_Throw.FORT)
                if other.dead:
                    before_hp = target.hp
                    other.hp = other.death_treshold
                    other.coma = True
                    t_weakest = all(target.hp <= other.hp for other in env.characters
                                    if self.plausible_target(actor, other) and not other.coma)
                    actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Petrification',
                                          target.name, target.fullDefense, t_weakest, -before_hp + other.hp,
                                          other.hp - before_hp])
                    return

        ac_to_hit = target.get_ac(True)
        t_weakest = all(
        target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        totalDamage = 0

        for bonus, dice, critMul in zip([11, 6], [Dice('1d8'), Dice('1d8')], [3, 3]):
            attack_roll, nat_roll = D20(bonus).roll()
            if (attack_roll >= ac_to_hit or nat_roll == 20) and nat_roll != 1:
                crit_hit = (nat_roll == 20) or target.unconscious or target.coma
                dmg_roll = dice.roll(crit_hit=crit_hit, crit_mul=critMul)
                totalDamage += dmg_roll

        if totalDamage != 0:
            before_hp = target.hp
            target.damage(totalDamage, actor.rangedWeapon.damage_type)
            after_hp = target.hp
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Medusa Ranged Attack',
                                  target.name, target.fullDefense, t_weakest, -totalDamage, after_hp - before_hp])
        else:
            # print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, 'Medusa Ranged Attack',
                                  target.name, target.fullDefense, t_weakest, 0, 0])

class RangedAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        ac_to_hit = target.get_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        # all([]) == True, which seems ok
        attack_roll, nat_roll = D20(actor.rangedToHit).roll()
        if (attack_roll >= ac_to_hit or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) <= actor.accuracy:
            crit_hit = (nat_roll >= actor.rangedWeapon.critRange) or target.unconscious or target.coma
            dmg_roll = actor.rangedDamageDie.roll(crit_hit=crit_hit, crit_mul=actor.rangedWeapon.critMul)
            before_hp = target.hp
            target.damage(dmg_roll, actor.rangedWeapon.damage_type)
            after_hp = target.hp
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.rangedWeapon.name,
                                  target.name, target.fullDefense, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.rangedWeapon.name,
                                  target.name, target.fullDefense, t_weakest, 0, 0])

class RangedDeadlyAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        ac_to_hit = target.get_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        # all([]) == True, which seems ok
        attack_roll, nat_roll = D20(actor.rangedToHit).roll()
        if (attack_roll-1 >= ac_to_hit or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) <= actor.accuracy:
            crit_hit = (nat_roll >= actor.rangedWeapon.critRange) or target.unconscious or target.coma
            dmg_roll = actor.rangedDSDamageDie.roll(crit_hit=crit_hit, crit_mul=actor.rangedWeapon.critMul)
            before_hp = target.hp
            target.damage(dmg_roll, actor.rangedWeapon.damage_type)
            after_hp = target.hp
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.rangedWeapon.name + ' Deadly Shot',
                                  target.name, target.fullDefense, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, actor.rangedWeapon.name,
                                  target.name, target.fullDefense, t_weakest, 0, 0])

class RapidShotRangedAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        ac_to_hit = target.get_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        totalDamage = 0

        for i in range(2):
            attack_roll, nat_roll = D20(actor.rangedToHit).roll()
            if (attack_roll - 2 >= ac_to_hit or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) > actor.accuracy:
                crit_hit = (nat_roll >= actor.rangedWeapon.critRange) or target.unconscious or target.coma
                dmg_roll = actor.rangedDamageDie.roll(crit_hit=crit_hit, crit_mul=actor.rangedWeapon.critMul)
                totalDamage += dmg_roll

        if totalDamage != 0:
            before_hp = target.hp
            target.damage(totalDamage, actor.rangedWeapon.damage_type)
            after_hp = target.hp
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, '2xRS '+actor.rangedWeapon.name,
                                  target.name, target.fullDefense, t_weakest, -totalDamage, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, '2xRS '+actor.rangedWeapon.name,
                                  target.name, target.fullDefense, t_weakest, 0, 0])

class RapidShotRangedDeadlyAttack(Action):
    def plausible_target(self, actor: Character, target: Character):
        return target.team != actor.team and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        ac_to_hit = target.get_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        totalDamage = 0

        for i in range(2):
            attack_roll, nat_roll = D20(actor.rangedToHit).roll()
            if (attack_roll - 3 >= ac_to_hit or nat_roll == 20) and nat_roll != 1 and random.uniform(0, 1) > actor.accuracy:
                crit_hit = (nat_roll >= actor.rangedWeapon.critRange) or target.unconscious or target.coma
                dmg_roll = actor.rangedDSDamageDie.roll(crit_hit=crit_hit, crit_mul=actor.rangedWeapon.critMul)
                totalDamage += dmg_roll

        if totalDamage != 0:
            before_hp = target.hp
            target.damage(totalDamage, actor.rangedWeapon.damage_type)
            after_hp = target.hp
            #print(f"{actor.name} attacked {target.name} with {self.name} for {dmg_roll}")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, '2xRS '+actor.rangedWeapon.name+ ' Deadly Shot',
                                  target.name, target.fullDefense, t_weakest, -totalDamage, after_hp - before_hp])
        else:
            #print(f"{actor.name} attacked {target.name} with {self.name} and missed")
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, '2xRS '+actor.rangedWeapon.name+ ' Deadly Shot',
                                  target.name, target.fullDefense, t_weakest, 0, 0])

class HealingPotion(Action):
    def __init__(self, name: str, heal_dice: str, uses: int = 1):
        self.name = name
        self.heal_dice = Dice(heal_dice)
        self.uses = uses

    def is_forbidden(self, actor: Character, env: Environment):
        return (self.uses <= 0)

    def plausible_target(self, actor: Character, target: Character):
        return target.team == actor.team and target.hp < target.max_hp and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        if self.is_forbidden(actor, env):
            return
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        # all([]) == True, which seems ok
        heal_roll = self.heal_dice.roll()
        before_hp = target.hp
        target.heal(heal_roll)
        after_hp = target.hp
        self.uses -= 1
        #print(f"{actor.name} used {self.name} on {target.name} for {heal_roll}")
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.fullDefense, t_weakest, heal_roll, after_hp - before_hp])

class Spell(Action):
    # def __init__(self, name: str, level: int, concentration: bool = False):
    #     self.name = name
    #     self.level = level # 0 for cantrips
    #     self.concentration = concentration

    def is_forbidden(self, actor: Character, env: Environment):
        return self.level > 0 and actor.curr_spells[self.level-1] <= 0

    def _consume_slot(self, actor: Character):
        if self.level > 0: actor.curr_spells[self.level-1] -= 1

    def _spell_attack(self, actor: Character, target: Character, env: Environment, dmg_dice: Dice, dmg_type: str):
        ac_to_hit = target.get_touch_ac(False)
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other) and not other.coma)
        # all([]) == True, which seems ok
        attack_roll, nat_roll = D20(actor.rangedToHit).roll()
        if (attack_roll >= ac_to_hit or nat_roll == 20) and nat_roll != 1:
            crit_hit = (nat_roll == 20) or target.unconscious # critical hits work with spell attacks too
            dmg_roll = dmg_dice.roll(crit_hit=crit_hit)
            before_hp = target.hp
            target.damage(dmg_roll, dmg_type)
            after_hp = target.hp
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.fullDefense, t_weakest, -dmg_roll, after_hp - before_hp])
        else:
            actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.fullDefense, t_weakest, 0, 0])

    def __call__(self, actor: Character, target: Character, env: Environment):
        if self.is_forbidden(actor, env): return
        self._consume_slot(actor)
        self.call(actor, target, env)

class MageArmor(Spell):
    name = 'Mage Armor'
    level = 1
    plausible_target = Action._conscious_ally

    def call(self, actor: Character, target: Character, env: Environment):
        target.ac = 10 + min(target.ability_mods[Ability.DEX], target.armor.maxDexBonus) + max(target.armor.acBonus, 4)
        target.flatFootedAc = 10 + max(target.armor.acBonus, 4)
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.fullDefense, False, 0, 0])

class ProtectionFromEvil(Spell):
    name = 'Protection From Evil'
    level = 1
    plausible_target = Action._unprotected_ally

    def call(self, actor: Character, target: Character, env: Environment):
        target.protected = True
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.fullDefense, False, 0, 0])

class CureLightWounds(Spell):
    name = 'Cure Light Wounds'
    level = 1
    plausible_target = Action._living_ally

    def __call__(self, actor: Character, target: Character, env: Environment):
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        heal_roll = Dice('1d8+2').roll()
        before_hp = target.hp
        target.heal(heal_roll)
        after_hp = target.hp
        #print(f"{actor.name} used {self.name} on {target.name} for {heal_roll}")
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.fullDefense, t_weakest, heal_roll, after_hp - before_hp])

class ChannelEnergy():
    def __init__(self, uses: int = 1):
        self.name = 'Channel Energy'
        self.heal_dice = Dice('1d6')
        self.uses = uses

    def is_forbidden(self, actor: Character, env: Environment):
        return (self.uses <= 0)

    def plausible_target(self, actor: Character, target: Character):
        return target.team == actor.team and target.hp < target.max_hp and not target.dead

    def __call__(self, actor: Character, target: Character, env: Environment):
        if self.is_forbidden(actor, env):
            return
        # t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        # all([]) == True, which seems ok
        heal_roll = self.heal_dice.roll()
        for target in env.characters :
            if self.plausible_target(actor, target):
                target.heal(heal_roll)
        self.uses -= 1
        #print(f"{actor.name} used {self.name} on {target.name} for {heal_roll}")
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, 'Team '+str(actor.team),
                              target.fullDefense, None, heal_roll, None])

class MagicMissle(Spell):
    name = 'Magic Missle'
    level = 1
    plausible_target = Action._conscious_enemy

    def call(self, actor: Character, target: Character, env: Environment):
        # Automatically hits.  TODO:  should be able to target multiple opponents
        t_weakest = all(target.hp <= other.hp for other in env.characters if self.plausible_target(actor, other))
        dmg_roll = roll('1d4+1')
        before_hp = target.hp
        target.damage(dmg_roll, 'force')
        after_hp = target.hp
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, target.name, target.fullDefense, t_weakest, -dmg_roll, after_hp - before_hp])

class RayOfFrost(Spell):
    name = 'Ray of Frost'
    level = 0 # cantrip
    plausible_target = Action._conscious_enemy

    def call(self, actor: Character, target: Character, env: Environment):
        self._spell_attack(actor, target, env, Dice('1d3'), 'cold')

class Sleep(Spell):
    name = 'Sleep'
    level = 1
    concentration = False
    plausible_target = Action._self_only
    # maybe this should be _conscious_enemy instead, if we factor out actions vs. targets?

    def call(self, actor: Character, target: Character, env: Environment):
        orig_hp = hp = roll('5d8')
        targets = [c for c in env.characters if c.team != actor.team and c.hp > 0 and not c.unconscious]
        targets.sort(key=lambda c: c.hp)
        t_names = []
        for target in targets:
            if target.hp <= hp:
                target.unconscious = True
                hp -= target.hp
                t_names.append(target.name)
            else:
                break
        actions_csv.writerow([epoch_id, encounter_id, round_id, actor.name, self.name, '/'.join(t_names), False, False, -orig_hp, -(orig_hp - hp)])

class Environment:
    def __init__(self, characters):
        assert len(characters) == MAX_CHARS
        self.characters = characters
        #self.battlefield = Graphttlefield(10, 10, self.characters)

    def run(self):
        chars = list(self.characters)
        # Initiative:
        rnd.shuffle(chars) # this ensures ties are broken randomly
        chars.sort(key=lambda c: D20(c.initiativeBonus).roll(), reverse=True) # key() is called once per item and cached
        global round_id
        round_id = 0
        while True:
            #print("== top of round ==")
            #print({c.name: c.hp for c in chars})
            for actor in chars:
                actor.start_of_round()
                if actor.dead or actor.coma or actor.unconscious:
                    continue
                actor.act(self)
            active_teams = set(c.team for c in chars if not c.coma)
            if len(active_teams) <= 1:
                break
            round_id += 1
        #print({c.name: c.hp for c in chars})
        for actor in chars:
            actor.end_of_encounter(self)
            outcomes_csv.writerow([
                epoch_id, encounter_id, round_id+1,
                actor.name, actor.team, (actor.team in active_teams),
                actor.max_hp, actor.hp
            ])
        return (0 in active_teams)

def init_workers(strats):
    #print(rnd.random()) # confirm that each process has unique random seed
    global strategies
    strategies = strats
    for s in strategies:
        s.alloc_buf()

def run_epoch(args):
    epoch_id_, n = args
    global strategies, epoch_id, encounter_id
    epoch_id = epoch_id_
    for s in strategies:
        s.buf.reset()

    '''
    Attacker : Fighter 2
    Feats selected (4) : Toughness, Weapon Focus, Improved Initiative, Power attack
    '''
    fighter_lvl2 = lambda i: PPOCharacter(strategies[0], name=f'Fighter {i}', team=0, hp=24, bab=2,
        armor=BreastPlate(),
        meleeWeapon=Greatsword(),
        rangedWeapon=Longbow(),
        actions=[
            MeleeAttack(),
            MeleePowerAttack(),
            RangedAttack(),
            FullDefense(),
            HealingPotion('potion of healing', '1d8+1', uses=2),
        ],
        ability_mods=[3,1,2,0,-1,-1], saving_throws=[5, 4, -1], initiative_boost=4, melee_boost=1)

    '''
    Defender : Fighter 2
    Feats selected (4) : Toughness, Shield focus, Armor Focus, Power attack
    '''
    defender_lvl2 = lambda i: PPOCharacter(strategies[1], name=f'Defender {i}', team=0, hp=26, bab=2,
      armor=BreastPlateAndShield2Focus(),
      meleeWeapon=Longsword(),
      rangedWeapon=None,
      actions=[
          MeleeAttack(),
          MeleePowerAttack(),
          FullDefense(),
          HealingPotion('potion of healing', '1d8+1', uses=1),
      ],
      ability_mods=[2, 1, 3, 0,-1,-1], saving_throws=[6, 4, -1])

    '''
    Archer : Fighter 2
    Feats selected (4) : Weapon Focus, Precise Shot, Point-blank Shot, Deadly Shot
    '''
    archer_lvl2 = lambda i: PPOCharacter(strategies[2], name=f'Archer {i}', team=0, hp=19, bab=2,
      armor=StuddedLeather(),
      meleeWeapon=Dagger(),
      rangedWeapon=Longbow(),
      actions=[
          MeleeAttack(),
          RangedAttack(),
          RangedDeadlyAttack(),
          RapidShotRangedAttack(),
          RapidShotRangedDeadlyAttack(),
          FullDefense(),
      ],
      ability_mods=[1, 3, 2, 1, 0, -1], saving_throws=[5, 6, 0], ranged_boost=2, initiative_boost=4, ranged_damage_boost=1)

    '''
    Healer : Cleric 2
    
    Feats : Toughness, Selective Channeling, Improved Initiative
    '''
    cleric_lvl2 = lambda i: PPOCharacter(strategies[3], name=f'Cleric {i}', team=0, hp=19, bab=1,
        armor=BreastPlateAndShield(),
        meleeWeapon=HeavyMace(),
        rangedWeapon=None,
        actions=[
            MeleeAttack(),
            FullDefense(),
            CureLightWounds(),
            ProtectionFromEvil(),
            ChannelEnergy(uses=3)
        ],
        ability_mods = [1, -1, 2, 0, 3, 1], saving_throws = [5, -1, 6],
        spells = [4, 0, 0, 0, 0, 0, 0, 0, 0], spell_save = 14,
        initiative_boost=4)


    medusa = lambda i: PPOCharacter(strategies[4], survival=0.5, name=f'Medusa {i}', team=1, hp=76, bab=8,
       armor=MedusaNaturalArmor(),
       meleeWeapon=Dagger(),
       rangedWeapon=Longbow(),
       actions=[
           UniqueMedusaMeleeAttack(),
           UniqueMedusaRangedAttack(),
           UniqueMedusaFullDefense()
       ],
       ability_mods=[0, 2, 4, 1, 1, 2], saving_throws=[6, 8, 7],
                                    ranged_boost=1, ranged_damage_boost=1, initiative_boost=4)

    '''
    Goblin Spider-Rider : Goblin Fighter 1 (CR1/2)
    URL : https://www.d20pfsrd.com/bestiary/npc-s/npcs-cr-0/goblin-spider-rider/
    
    Feats : Improved Initiative, Quick Draw (No effect in simulation)
    '''
    goblin_fighter = lambda i: PPOCharacter(strategies[1], name=f'Goblin {i}', team=1, hp=6, bab=1,
        armor=SmallScaleMail(),
        meleeWeapon=SmallLongspear(),
        rangedWeapon=SmallShortbow(),
        actions=[
            MeleeAttack(),
            RangedAttack(),
            FullDefense(),
        ],
        ability_mods=[1,1,1,0,2,0], saving_throws=[3, 1, 2], initiative_boost=4)

    '''
    Zombie : Medium Undead (CR1/2)
    URL : https://www.d20pfsrd.com/bestiary/monster-listings/undead/zombie
    
    Feats : Toughness
    '''
    zombie = lambda i: PPOCharacter(strategies[1], name=f'Zombie {i}', team=1, hp=12, bab=0,
        armor=ZombieNaturalArmor(),
        meleeWeapon=Slam(),
        rangedWeapon=None,
        actions=[
            MeleeAttack(),
            FullDefense(),
        ],
        ability_mods=[4, 0, 0, 0, 0, 0], saving_throws=[0, 0, 3])

    '''
    Giant Crab Spider
    URL : https://www.d20pfsrd.com/bestiary/monster-listings/vermin/spider/spider-giant-crab
    
    Feats : None
    '''
    spider = lambda i: PPOCharacter(strategies[1], name=f'Spider {i}', team=1, hp=11, bab=1,
        armor=SpiderNaturalArmor(),
        meleeWeapon=Spiderbite(),
        rangedWeapon=None,
        actions=[
            SpiderBite(),
            FullDefense(),
        ],
        ability_mods=[0, 2, 1, 0, 0, -4], saving_throws=[4, 2, 0])

    wins = 0
    for encounter_id in range(n):
        env = [fighter_lvl2(1), defender_lvl2(1), archer_lvl2(1), cleric_lvl2(1), medusa(1)]
        #for i in range(4): env.append(spider(i+1))
        env = Environment(env)
        if env.run(): wins += 1
    return [s.buf for s in strategies] + [wins]

def run_update(args):
    ppo_buffers, strategy = args
    data = merge_ppo_data(ppo_buffers)
    strategy.update(data)

def merge_ppo_data(ppo_buffers):
    data = [x.get() for x in ppo_buffers]
    out = {}
    for key in data[0].keys():
        out[key] = torch.cat([x[key] for x in data])
    return out

def main(epochs, ncpu):
    epochs = epochs
    ncpu = ncpu # using 8 doesn't seem to help on an M1
    strategies = [PPOStrategy(5), PPOStrategy(4), PPOStrategy(6), PPOStrategy(5), PPOStrategy(3)]
    with multiprocessing.Pool(ncpu, init_workers, (strategies,)) as pool:
        for epoch in trange(epochs):
            t1 = time.time()
            results = pool.map(run_epoch, [(epoch, 1000//ncpu) for _ in range(ncpu)])
            # transpose results matrix so entries of same type are together
            results = list(zip(*results))
            t2 = time.time()
            wins = np.array(results[-1]) * ncpu
            pool.map(run_update, [(results[i], s) for i, s in enumerate(strategies)])
            t3 = time.time()
            print(f"Epoch {epoch:04d}:  {wins.mean():.0f} ± {wins.std():.0f} wins in {t2-t1:.1f} + {t3-t2:.1f} sec")

if __name__ == '__main__':
    main()
