from rich.console import Console


from utils import ModelGenerator, DataCollector


def main():
    console = Console()
    model_generator = ModelGenerator()
    data_collector = DataCollector()
    with console.status("[bold green] Updating models...") as status:
        data_collector.update()
        console.log("[cyan] Successfully updated data")
        model_generator.update()
        console.log("[cyan] Successfully updated models")



if __name__ == '__main__':
    main()