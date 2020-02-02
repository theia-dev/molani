from textwrap import dedent as d

new_mol = "mol new \"{path}\"\n"

display_setting = d('''\
                    # set up display
                    axes location off
                    display projection orthographic
                    display resize {x} {y}
                    display projection {projection}
                    display shadows {shadows}
                    display depthcue {depthcue}
                    display antialias {antialias}
                    display ambientocclusion {ao}
                    display aoambient {aoambient}
                    display aodirect {aodirect}
                    display nearclip set {nearclip}
                    \n''')

color_background = d('''\
                    # set background color
                    color change rgb 0 {r} {g} {b}
                    color Display Background 0
                    \n''')

color_foreground = d('''\
                    # set foreground color
                    color change rgb 1 {r} {g} {b}
                    color Display Foreground 1
                    \n''')

set_color = "color change rgb {id} {r} {g} {b}\n"

play_script = 'play {path}\n'

mol_del_style = d("mol delrep {rid} top\n")

mol_style_base = d('''\
                   # set up mol
                   mol addrep top
                   mol modselect {rid} top {{{select}}}
                   mol modstyle {rid} top {style}
                   mol modmaterial {rid} top {material}
                   \n''')

mol_style_color_ID = "mol modcolor {rid} top ColorID {cid}\n"

mol_style_color_name = d('''\
                      mol modcolor {rid} top {name}
                      mol scaleminmax top {rid} {min} {max}
                      \n''')

mol_style_periodic = 'mol showperiodic top {rid} "{periodic}"\n'

value_names = ['"user"', '"user2"', '"user3"', '"user4"', 'beta']
value_names_system = ['"mass"', '"charge"']
value_display_names = {'"user"': '"user"', '"user2"': '"user2"', '"user3"': '"user3"', '"user4"': '"user4"',
                       'beta': 'beta', '"mass"': '"Mass"', '"charge"': '"Charge"'}
set_value = "${selection_name} set {name} {value}\n"
create_selection = "set {selection_name} [atomselect top {{{select}}}]\n"


add_trajectory = "mol addfile {path} first {first} last {last} waitfor -1\n"

scale = "scale to {scale}\n"
rotate_to = "rotate {axis} to {angle}\n"
rotate_by = "rotate {axis} by {angle}\n"

load_frame = "animate goto {frame}\n"
render_intern = "render TachyonInternal {path}\n"
render = 'render Tachyon "{temp_path}" \\"{tachyon}\\" -aasamples 4 -trans_vmd -fullshade -numthreads {cpu} -format TARGA "{temp_path}" -o "{path}"\n'

vmd_exit = 'exit\n\n'




