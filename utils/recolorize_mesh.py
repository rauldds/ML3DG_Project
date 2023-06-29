import trimesh

#BASED ON: https://github.com/mikedh/trimesh/issues/865

def colorizing(colored_mesh,recon_mesh):
    # use the same material as the fuze bottle
    #print(colored_mesh.visual.vertex_colors.shape)
    colored_mesh.visual = colored_mesh.visual.to_texture()
    recon_mesh.visual = colored_mesh.visual.copy()
    # query the nearest vertex index from original mesh
    v_idx = colored_mesh.kdtree.query(recon_mesh.vertices.copy())[1]
    #set the UV coordinates on the new mesh to the UV
    #coordinates of the nearest point from the original
    recon_mesh.visual.uv = colored_mesh.visual.uv[v_idx]
    recon_mesh.visual = recon_mesh.visual.to_color()
    #print(recon_mesh.visual.vertex_colors.shape)

    return recon_mesh

