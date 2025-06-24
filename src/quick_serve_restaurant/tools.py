


def insert_coffee_order_tool(
  host: CompositeVariableModel | dict[str, Any],
  token: CompositeVariableModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    if isinstance(host, dict):
        host = CompositeVariableModel(**host)
    if isinstance(token, dict):
        token = CompositeVariableModel(**token)

    @tool
    def insert_coffee_order(coffee_name: str, size: str, session_id: str) -> tuple:
      """
      Here is great description

      Args:
        coffee_name (str): The name of the coffee.
        size (str): The size of the coffee.
        session_id (str): The session ID of the user
      """


      uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)

      insert_tool = uc_toolkit.tools[0]
      params_dict = {
        "host": params["host"],
        "token": params["token"],
        "coffee_name": coffee_name,
        "size": size,
        "session_id": session_id,
      }
      return insert_tool(params_dict)


    return insert_coffee_order