from homeassistant_api import Client
import time
homeass = "http://homeassistant.home:8123/"
homeass_token = "homeass_long_lived_access_token"

haclient = Client(homeass, homeass_token)

# Replace "switch.your_switch_entity_id" with the entity ID of your switch
switch_entity_id = "switch.your_switch_entity_id"

# Turn on the switch
haclient.call_service("switch", "turn_on", {"entity_id": switch_entity_id})
time.sleep(10)
# Turn off the switch
haclient.call_service("switch", "turn_off", {"entity_id": switch_entity_id})