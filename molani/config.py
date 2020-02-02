from pathlib import Path

app_name = 'molani'
version_number = (0, 0, 1)
version = f'{version_number[0]}.{version_number[1]}.{version_number[2]}'
app_author = 'Hagen Eckert'
url = 'https://github.com/theia-dev/molani'

base_dir = Path(__file__).absolute().parent
data_folder = base_dir / 'data'
