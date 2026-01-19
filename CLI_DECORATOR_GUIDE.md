# CLI Registration System

The CLI now uses a decorator-based registration system. This allows you to create new commands without modifying `cli.py`.

## How to Create a New Command

1. Create a new Python file in `scripts/ac/` or `scripts/car-racing/`
2. Import the decorators:
   ```python
   from assetto_corsa_rl.cli_registry import cli_command, cli_option
   ```

3. Decorate your main function:
   ```python
   @cli_command(
       group="ac", 
       name="my-command",  
       help="Description of what this command does"
   )
   @cli_option("--input", required=True, help="Input file path")
   @cli_option("--epochs", default=100, type=int, help="Number of epochs")
   @cli_option("--verbose", is_flag=True, help="Enable verbose mode")
   def main(input, epochs, verbose):
       # Your code here
       print(f"Running with {epochs} epochs")
   ```

4. Your command is now available:
   ```bash
   acrl ac my-command --input data.txt --epochs 50 --verbose
   ```

## Example

See `scripts/ac/example_command.py` for a complete example.

## Benefits

- **No manual CLI updates**: Just create a file with decorators
- **Auto-discovery**: The CLI finds your script automatically  
- **Type-safe**: Click handles argument parsing and validation
- **Self-documenting**: Help text comes from your decorators
- **Modular**: Each script is independent and can still be run standalone

## Decorator Reference

### `@cli_command(group, name=None, help=None, short_help=None)`

Registers a function as a CLI command.

- `group`: Command group ("ac" or "car-racing")
- `name`: Command name (defaults to function name)
- `help`: Full help text
- `short_help`: Brief description for listings

### `@cli_option(*param_decls, **attrs)`

Adds a CLI option/flag to the command.

Common attributes:
- `required=True`: Make option required
- `default=value`: Default value
- `type=int|float|str`: Value type
- `is_flag=True`: Boolean flag (no value needed)
- `help="text"`: Help text for this option

Multiple names: `@cli_option("--input", "-i", ...)`

## Migration Guide

Old (hardcoded in cli.py):
```python
@ac.command()
@click.option("--input", required=True)
def my_command(input):
    from ac import my_script
    my_script.main(input)
```

New (in your script file):
```python
@cli_command(group="ac", name="my-command")
@cli_option("--input", required=True)
def main(input):
    # Your code directly here
    pass
```
