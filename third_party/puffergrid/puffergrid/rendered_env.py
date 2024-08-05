
        self.render_mode = render_mode
        self.renderer = render.make_renderer(map_size, map_size,
            render_mode=render_mode, sprite_sheet_path='./env/mettagrid/puffer-128-sprites.png')

def render(self, mode='human'):
    return self.renderer.render(self.grid[1, :, :] !=0 ,
        self.agent_positions, self.actions, self.vision_range)
