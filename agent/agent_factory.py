from agent.object_attn_agent import ObjectAttnAgent
from agent.object_embedding_agent import ObjectEmeddingAgent, ObjectEmeddingAgentDecoder


agent_specs = {
    "object_embedding": {

    }
}

class AgentFactory():
    def create_agent(self, agent_spec_id):
        if agent_spec_id == "object_attn_agent":
            return ObjectAttnAgent()
        if agent_spec_id == "object_embedding_agent":
            return ObjectEmeddingAgent()
        raise ValueError(f"Unknown agent spec {agent_spec_id}")
