from fastapi import FastAPI
from env import EmailEnv, Action

app = FastAPI()
# Create one global instance of the Post Office
office = EmailEnv(num_emails=10)

@app.get("/health")
def health():
    """Tells the judges the server is awake."""
    return {"status": "ok"}

@app.post("/reset")
def reset():
    """The automated check trying to reset the environment."""
    obs = office.reset()
    return obs.model_dump()

@app.get("/state")
def state():
    obs = office.state()
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    """The automated check trying to take an action."""
    obs, reward, done, info = office.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info
    }