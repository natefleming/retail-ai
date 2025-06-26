create or replace function {catalog_name}.{schema_name}.insert_coffee_order(
  host string comment "Databricks workspace host URL - automatically provided by the system context. Do not ask customers for this value."
, token string comment "Databricks authentication token - automatically provided by the system context. Do not ask customers for this value."
, coffee_name string comment "Exact name of the coffee item being ordered. Examples: 'Cappuccino', 'Iced Latte', 'Cold Brew', 'Caramel Macchiato'. Use the exact menu item name from previous searches or customer specifications."
, size string comment "Size of the coffee order. Valid options: 'Small', 'Medium', 'Large', or 'N/A' for single-size items. Always confirm size with customer before placing order."
, session_id string comment "Unique session identifier for this customer conversation - automatically provided by the system as thread_id. Do not ask customers for this value."
) 
returns string
language python 
COMMENT 'Process and record a coffee order in the fulfillment system. Use this tool ONLY when a customer explicitly wants to place an order with clear ordering language like "I want to order", "I will take", "Can I get", or "I would like". Always confirm the coffee name and size before calling this tool. This creates an actual order record that will be fulfilled, so only use when the customer is ready to purchase. Returns order confirmation message indicating success ("Row successfully inserted - SUCCEEDED") or error message if order failed.'
AS 
$$
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Format, Disposition
import uuid
def get_sql_warehouse(w):
    l_wh_id = [wh.warehouse_id for wh in w.data_sources.list() if 'Shared Endpoint' in wh.name]
    return l_wh_id[0]

def run_sql_statement(w, statement: str):
    wh_id = get_sql_warehouse(w)
    print(wh_id)
    
    statement_execute_response_dict = w.statement_execution.execute_statement(warehouse_id=wh_id
                                                                              , format=Format.JSON_ARRAY
                                                                              , disposition=Disposition.INLINE
                                                                              , statement=statement
                                                                              ).as_dict()
    return statement_execute_response_dict["status"]['state']

w = WorkspaceClient(host=host, token=token)
uuid = str(uuid.uuid4())
uuid = f"'{uuid}'"
statement = f"insert into {catalog_name}.{schema_name}.fulfil_item_orders(uuid, coffee_name, size, session_id) values ({uuid}, '{coffee_name}', '{size}', '{session_id}')"
response=run_sql_statement(w, statement)
if response == 'SUCCEEDED':
  return f"Row successfully inserted - {response}"
else:
  return f"Error inserting row - {response}"
$$