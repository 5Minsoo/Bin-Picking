# 🤖 KITECH Bin-Picking: Gripper Grip Detection

![Generic badge](https://img.shields.io/badge/Project-KITECH-blue.svg)
![Generic badge](https://img.shields.io/badge/Tech-Isaac__Sim-green.svg)
![Generic badge](https://img.shields.io/badge/Type-Bin__Picking-orange.svg)

**한국생산기술연구원(KITECH)** Bin Picking 과제: 파이프 배관(Flange Pipe) 파지 및 그립 성공 여부 판단 프로젝트

## 📌 Overview
본 프로젝트는 **Isaac Sim** 기반의 합성 데이터와 **강화 학습(Reinforcement Learning)**을 활용하여 비정형 환경에 놓인 배관 부품(Flange)을 안정적으로 파지하는 것을 목표로 합니다.

- **Target Object**: 플랜지(Flange) 형상의 배관 부품
- **Core Technology**: Isaac Sim (Sim-to-Real), Reinforcement Learning
- **Goal**: Vision 기반 그립 최적화 및 Sensor Feedback 기반 파지 성공 판단

<br/>

## ⚙️ System Logic (Flowchart)

```mermaid
graph TD;
    A[📷 Camera Input] -->|Object Recognition| B(Pose Estimation);
    B -->|Reinforcement Learning| C[🎯 Grip Optimization];
    C --> D[🤖 Execute Grip];
    D --> E{Grip Detection};
    E -- Success (Joint/Force OK) --> F[✅ Transport];
    E -- Fail (Slip/Miss) --> G[🔄 Retry Strategy];
    G --> D;
