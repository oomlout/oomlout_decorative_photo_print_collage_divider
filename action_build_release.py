import os

def main(**kwargs):
    # load configuration file
    file_configuration = "configuration/generate_release.yaml"
    import yaml
    with open(file_configuration, 'r') as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    repos = configuration["repo"]

    #clone all repos into temporary directory
    for repo in repos:
        url = repo
        #add https if not there
        if not url.startswith("https://"):
            url = f"https://{url}"
        repo_dir = url.split("/")[-1]
        repo_dir = f"temporary\\{repo_dir}"
        if not os.path.exists(repo_dir):            
            os.system(f"git clone {url}.git {repo_dir}")
        else:
            os.system(f"cd {repo_dir} && git pull")

    #   copy the files to release directory
    files_to_copy = configuration["file"]
    ## hinged version
    
    for file in files_to_copy:
        source = file["source"]
        repo = file["repo"]
        destination = file["destination"]

        source_full = f"temporary\\{repo}\\{source}"

        destination = file["destination"]
        #make destination directory
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        #if source exists delete it
        if os.path.exists(destination):
            os.remove(destination)
        command = f"copy {source_full} {destination}"
        print(f"Copying {source_full} to {destination}")        
        os.system(command)
        
        #if ends in stl
        if source_full.endswith(".stl"):
            #try to copy scad and png too
            versions = [".scad",".png"]
            for version in versions:
                source_full_2 = source_full.replace(".stl",version)
                destination_2 = destination.replace(".stl",version)
                if os.path.exists(source_full_2):
                        command = f"copy {source_full_2} {destination_2}"
                        print(f"Copying {source_full_2} to {destination_2}")
                        os.system(command)
            



if __name__ == '__main__':
    kwargs = {}
    main(**kwargs)