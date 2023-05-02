
import argparse


class Toolkit():
    def __init__(self, tools):
        self.tools = tools


    def list_tools(self):
        return self.tools

    def list_tools_print(self):
        for tool in self.list_tools():
            print(f"\n\n{tool.name}")
            print(f"\t{tool.description}")

    def run_cli(self):
        parser = argparse.ArgumentParser(description='Run a statistical tool')
        parser.add_argument('--tool', type=str, help='the name of the tool to run')
        parser.add_argument('--args', type=str, help='comma separated list of arguments to pass to the tool')
        parser.add_argument('--list', action='store_true', help='list all available tools')
        args = parser.parse_args()
        if args.list:
            self.list_tools_print()
        else:
            tool = next(filter(lambda x: x.name == args.tool, self.list_tools()))
            print(tool(args.args))
