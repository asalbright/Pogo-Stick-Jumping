# Description of Training Data - 2022/01/18
This is a short description of the training data that is within this repository. It is meant to act as a history marker for tracking testing of different training parameters including things like hyperparameter selection.

## Training
Agents were trained according to the `training.py` script in the repository. They were trained with a randomly placed actuator position and static spring constant and damping ration value. See table for parameters.

| Spring Constant Range |  Zeta Range | Actuator Position |
|:---------------------:| :----------:|-------------------|
|     2038 to 7030      | 0.0 to 0.01 | 0 : Max Act Pos   |

## Evaluation
Agents were evaluated on **three** sets of environment parameters with the actuator position starting at mid-stroke. See table for parameters. The data for the performance is within `/figures_data/` and `/figures` folders.

|  Spring Constants | Zetas | Actuator Position |
|:-----------------:|:-----:|-------------------|
|  3760, 5760, 7760 |  0.01 | 0.5*(Max Act Pos) |

## File Naming Convention

The naming convention within the `/figures_data/` is as follows: 
`/xxx_yyy/`
- `xxx` = spring constant tested
- `yyy` = damping ratio tested

The naming convention within the `/xxx_yyy/` is as follows:
`aaa_bbb_ccc_ddd_eee_fff.csv`
- `aaa` = Step during training the agent was generated
- `bbb` = Type of agent that was evaluated
  - Effic = Efficient Agent
  - Heigh = Height Agent
  - SpecH = Specified Height Agent
  - SpHiEf = Specified Height Efficiently Agent
- `ccc` = Type of jump the agent was trained to accomplish
- `ddd` = The network initialization seed
- `eee` = Date stamp on data
- `fff` = Time stamp on data

Additionally there is a `/Combined_Data` folder within `/xxx_yyy/`. This is simply a compilation of all the data within the `/xxx_yyy/` folder which may make plotting averages and standard deviations easier. 

## Data at A Glance
### Spring Constant: 3760
#### Height Agent
<p align="center">
    <img width="75%" src="Heigh_StutterJump_robust/figures/3760_0.01_Input_20220118_180720.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="Heigh_StutterJump_robust/figures/3760_0.01_RodPos_20220118_180722.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

#### Efficient Agent
<p align="center">
    <img width="75%" src="Effic_StutterJump_robust/figures/3760_0.01_Input_20220118_180552.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="Effic_StutterJump_robust/figures/3760_0.01_RodPos_20220118_180555.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

#### Specified Height Agent
<p align="center">
    <img width="75%" src="SpecH_StutterJump_robust/figures/3760_0.01_Input_20220118_180822.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="SpecH_StutterJump_robust/figures/3760_0.01_RodPos_20220118_180824.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

### Spring Constant: 5760
#### Height Agent
<p align="center">
    <img width="75%" src="Heigh_StutterJump_robust/figures/5760_0.01_Input_20220118_180736.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="Heigh_StutterJump_robust/figures/5760_0.01_RodPos_20220118_180737.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

#### Efficient Agent
<p align="center">
    <img width="75%" src="Effic_StutterJump_robust/figures/5760_0.01_Input_20220118_180614.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="Effic_StutterJump_robust/figures/5760_0.01_RodPos_20220118_180616.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

#### Specified Height Agent
<p align="center">
    <img width="75%" src="SpecH_StutterJump_robust/figures/5760_0.01_Input_20220118_180839.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="SpecH_StutterJump_robust/figures/5760_0.01_RodPos_20220118_180840.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

### Spring Constant: 7760
#### Height Agent
<p align="center">
    <img width="75%" src="Heigh_StutterJump_robust/figures/7760_0.01_Input_20220118_180748.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="Heigh_StutterJump_robust/figures/7760_0.01_RodPos_20220118_180749.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

#### Efficient Agent
<p align="center">
    <img width="75%" src="Effic_StutterJump_robust/figures/7760_0.01_Input_20220118_180630.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="Effic_StutterJump_robust/figures/7760_0.01_RodPos_20220118_180631.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>

#### Specified Height Agent
<p align="center">
    <img width="75%" src="SpecH_StutterJump_robust/figures/7760_0.01_Input_20220118_180852.png" alt="System Input"/><br>
    <strong>System Input</strong>
</p>

<p align="center">
    <img width="75%" src="SpecH_StutterJump_robust/figures/7760_0.01_RodPos_20220118_180854.png" alt="System Input"/><br>
    <strong>Pogo Height</strong>
</p>