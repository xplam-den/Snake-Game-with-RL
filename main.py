import Env
import Agent
            
Env = Env.SnakeEnv(20, 20)
Agent = Agent.SnakeAgent(Env,
                        Hv=5,
                        Wv=5,
                        set_kernel=False,
                        nEpisode=888,
                        mStep=128,
                        sMemory=1024,
                        sBatch=64,
                        TrainOn=32,
                        Gamma=0.95,
                        Alpha1=0.95,
                        Alpha2=0.65,
                        Temperature1=15,
                        Temperature2=0.5,
                        savemodel=True,
                        Load=False
                        )

Agent.CreateModel(nDense=[512,256], Activation='selu')
print(20*'-'+'Model Created.')

Agent.CompileModel()
print(20*'-'+'Model Compiled.')

Agent.SummaryModel()
print(20*'-')

# Agent.PlotEpsilons()
# print(20*'-')

Agent.Train(Policy='B')
print(20*'-'+'Trained!')

Agent.Test('G', 5)
print(20*'-','ebded')
