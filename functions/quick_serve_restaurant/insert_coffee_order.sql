create or replace function {catalog_name}.{schema_name}.insert_coffee_order(
  host string comment "The host of the Databricks workspace"
, token string comment "The token of the pricipal of the Databricks workspace"
, coffee_name string comment "The name of the coffee to be ordered"
, size string comment "The size of the coffee to be ordered"
, session_id string comment "The session_id of the user"
) 
returns string 
language python 
COMMENT 'The function is used as a tool to insert record into the order fulfillment table in the retail. Get the host and token from the l;ocal context. Return success once you are able to successfully insert into the table.'
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
statement = f"insert into retail.coffeeshop.fulfil_item_orders_jh (uuid, coffee_name, size, session_id) values ({uuid}, '{coffee_name}', '{size}', '{session_id}')"
response=run_sql_statement(w, statement)
if response == 'SUCCEEDED':
  return f"Row successfully inserted - {response}"
else:
  return f"Error inserting row - {response}"
$$