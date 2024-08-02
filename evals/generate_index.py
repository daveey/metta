import os
from pathlib import Path
from collections import defaultdict
import re
import sys

def generate_html(files_by_category):
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Behavior Simulations Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            text-align: center;
            color: #2c3e50;
        }
        .category {
            margin-bottom: 40px;
        }
        .simulation {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }
        .simulation h3 {
            color: #3498db;
        }
        img, video {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px 0;
        }
        a {
            color: #e74c3c;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Behavior Simulations Gallery</h1>
    """

    # Custom order for categories
    category_order = ['complex', 'empty', 'altar', 'resources', 'resource competition', 'resource movement cost', 'other']

    for category in category_order:
        if category not in files_by_category:
            continue

        files = files_by_category[category]
        html_content += f"""
    <div class="category">
        <h2>{category.title()} Simulations</h2>
        """

        for file in files:
            base_name = file.stem
            title = base_name.replace('_', ' ').title()
            description = generate_description(base_name)

            html_content += f"""
        <div class="simulation">
            <h3>{title}</h3>
            """

            if file.suffix == '.mp4':
                html_content += f"""
            <video width="100%" controls>
                <source src="{file.name}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
                """
            elif file.suffix == '.gif' and not (file.with_suffix('.mp4') in files):
                html_content += f'<img src="{file.name}" alt="{title}">'

            html_content += f"""
            <p>{description}</p>
        </div>
        """

        html_content += """
    </div>
        """

    html_content += """
</body>
</html>
    """
    return html_content

def generate_description(file_name):
    parts = file_name.split('_')
    if 'empty' in parts:
        return f"Agents in an empty room."
    elif 'a' in parts[0] and parts[0][1:].isdigit():
        agents = parts[0][1:]
        size = parts[-1]
        return f"{agents} agents in a complex {size} dimensional environment."
    elif 'altar' in parts:
        return f"A {parts[-1]} dimensional room simulating altar behavior."
    elif 'resources' in parts:
        if 'competition' in parts:
            agents = '2' if '2a' in parts else '3'
            return f"A {parts[-1]} dimensional room simulating resource competition between {agents} agents."
        elif 'move' in parts and 'cost' in parts:
            return f"A {parts[-1]} dimensional room simulating resource movement with associated costs."
        else:
            return f"A {parts[-1]} dimensional room with resource distribution."
    else:
        return f"A simulation of {' '.join(parts)} behavior."

def get_map_size(file_name):
    match = re.search(r'(\d+)x(\d+)', file_name)
    if match:
        return int(match.group(1)) * int(match.group(2))
    return 0

def categorize_simulations(files):
    categories = defaultdict(list)
    for file in files:
        if 'empty' in file.stem:
            categories['empty'].append(file)
        elif file.stem[0] == 'a' and file.stem[1:].split('_')[0].isdigit():
            categories['complex'].append(file)
        elif 'altar' in file.stem:
            categories['altar'].append(file)
        elif 'resources' in file.stem:
            if 'competition' in file.stem:
                categories['resource competition'].append(file)
            elif 'move' in file.stem and 'cost' in file.stem:
                categories['resource movement cost'].append(file)
            else:
                categories['resources'].append(file)
        else:
            categories['other'].append(file)

    # Sort each category by map size
    for category in categories:
        categories[category].sort(key=lambda x: get_map_size(x.stem))

    return categories

def main(directory):
    directory_path = Path(directory)
    if not directory_path.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    files = list(directory_path.glob('*.gif')) + list(directory_path.glob('*.mp4'))

    if not files:
        print(f"No GIF or MP4 files found in the directory: {directory}")
        return

    categorized_files = categorize_simulations(files)
    html_content = generate_html(categorized_files)

    output_file = directory_path / 'index.html'
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"open {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    main(directory)
