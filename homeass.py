from homeassistant_api import Client,State
import asyncio
import time
homeass = "http://localhost:8123/api/"
homeass_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIzODM2ODY2OWNhNTQ0MGRlODk5ODA5NGJjZmJiMjMyNiIsImlhdCI6MTcxODA5NjQ2OSwiZXhwIjoyMDMzNDU2NDY5fQ.V_3CZP7ZcR5eHNHAJmUOJlh-9wOQoF8ZzXxmSd5Mz_8"

client = Client(homeass, homeass_token, use_async=True)

async def main():
    #entity_groups = await client.async_get_entities()
    #sw = await client.async_get_entity(entity_id='input_boolean.test_switch')
    #states = await client.async_get_states()
    #state = await client.async_get_state(entity_id='input_boolean.test_switch')
    state = await client.async_get_state(entity_id='input_boolean.test_switch')
    
    new_state = await client.async_set_state(State(entity_id='input_boolean.test_switch', state='off'))
    state2 = await client.async_get_state(entity_id='input_boolean.test_switch')
  
    print(state)
    
    
  
asyncio.get_event_loop().run_until_complete(main())
asyncio.get_event_loop().run_until_complete(main())