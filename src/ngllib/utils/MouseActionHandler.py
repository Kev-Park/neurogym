import time

class MouseActionHandler:
    def __init__(self, page):
        self.page = page

    def keys_pressed(self, keysPressed):
        """Handles key states (Shift, Ctrl, Alt)"""
        key_down_script = ""
        if 'Shift' in keysPressed:
            key_down_script += """
                var shiftDown = new KeyboardEvent('keydown', {
                    key: 'Shift',
                    code: 'ShiftLeft',
                    keyCode: 16,
                    bubbles: true,
                    cancelable: true
                });
                window.dispatchEvent(shiftDown);
            """
        if 'Ctrl' in keysPressed:
            key_down_script += """
                var ctrlDown = new KeyboardEvent('keydown', {
                    key: 'Control',
                    code: 'ControlLeft',
                    keyCode: 17,
                    bubbles: true,
                    cancelable: true
                });
                window.dispatchEvent(ctrlDown);
            """
        if 'Alt' in keysPressed:
            key_down_script += """
                var altDown = new KeyboardEvent('keydown', {
                    key: 'Alt',
                    code: 'AltLeft',
                    keyCode: 18,
                    bubbles: true,
                    cancelable: true
                });
                window.dispatchEvent(altDown);
            """
        return key_down_script

    def clear_keys_pressed(self, keysPressed):
        """Clears key states (Shift, Ctrl, Alt) by dispatching keyup events."""
        key_up_script = ""
        if 'Shift' in keysPressed:
            key_up_script += """
                var shiftUp = new KeyboardEvent('keyup', {
                    key: 'Shift',
                    code: 'ShiftLeft',
                    keyCode: 16,
                    bubbles: true,
                    cancelable: true
                });
                window.dispatchEvent(shiftUp);
            """
        if 'Ctrl' in keysPressed:
            key_up_script += """
                var ctrlUp = new KeyboardEvent('keyup', {
                    key: 'Control',
                    code: 'ControlLeft',
                    keyCode: 17,
                    bubbles: true,
                    cancelable: true
                });
                window.dispatchEvent(ctrlUp);
            """
        if 'Alt' in keysPressed:
            key_up_script += """
                var altUp = new KeyboardEvent('keyup', {
                    key: 'Alt',
                    code: 'AltLeft',
                    keyCode: 18,
                    bubbles: true,
                    cancelable: true
                });
                window.dispatchEvent(altUp);
            """
        return key_up_script

    def execute_click(self, x: float, y: float, action: str, keysPressed:str = "None", grid_size: float = 25):
        """Perform a mouse action on the neuroglancer viewer
        Args:
        x: float: x-coordinate of the action, horizontal metric on the browser (like width). The action is relative to the size of the rendered part of the viewer
        y: float: y coordinate, vertical metric
        action: str: action to perform, one of "left_click", "double_click", "move", "right_click" (which is a context click)
        keysPressed: str: keys pressed during the action, separated by commas
        """
        keys = []
        element_rect = self.page.locator('.neuroglancer-layer-group-viewer').bounding_box()
        clamped_x = max(0, min(x, element_rect['width'] - 1))
        clamped_y = max(0, min(y, element_rect['height'] - 1))

        key_down_script = self.keys_pressed(keys)
        key_up_script = self.clear_keys_pressed(keysPressed)

        js_action_script = f"""
            var eventType = "{'dblclick' if action == 'double_click' else 'contextmenu' if action == 'right_click' else 'click'}";
            var element = document.querySelector('.neuroglancer-layer-group-viewer');

            if (!element) {{
                console.error("Element '.neuroglancer-layer-group-viewer' not found");
            }} else {{
                const rect = element.getBoundingClientRect();
                const targetX = rect.left + {clamped_x};
                const targetY = rect.top + {clamped_y};

                var childElement = document.elementFromPoint(targetX, targetY);

                if (childElement && element.contains(childElement)) {{
                    {key_down_script}
                    // Dispatch mousedown, main event, and mouseup sequentially on the child element
                    var mousedownEvent = new MouseEvent('mousedown', {{
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: targetX,
                        clientY: targetY,
                        button: {2 if action == 'right_click' else 0}
                    }});

                    childElement.dispatchEvent(mousedownEvent);

                    var mainEvent = new MouseEvent(eventType, {{
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: targetX,
                        clientY: targetY,
                        button: {2 if action == 'right_click' else 0}
                    }});

                    childElement.dispatchEvent(mainEvent);

                    var mouseupEvent = new MouseEvent('mouseup', {{
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: targetX,
                        clientY: targetY,
                        button: {2 if action == 'right_click' else 0}
                    }});

                    childElement.dispatchEvent(mouseupEvent);
                }} else {{
                    console.error("No child element found at the specified position or the child is outside the target area.");
                }}
            }}
        """

        self.page.evaluate(f"() => {{\n{js_action_script}\n}}")

    def add_visual_marker(self, clamped_x, clamped_y, element_rect):
        js_marker_script = f"""
            var markers = document.querySelectorAll('.selenium-pointer');
            markers.forEach(function(marker) {{
                marker.remove();
            }});

            var marker = document.createElement('img');
            marker.src = 'https://img.icons8.com/color/48/cursor--v1.png';
            marker.style.position = 'absolute';
            marker.style.left = '{clamped_x + element_rect['x']}px';
            marker.style.top = '{clamped_y + element_rect['y']}px';
            marker.style.width = '24px';
            marker.style.height = '24px';
            marker.className = 'selenium-pointer';
            document.body.appendChild(marker);
        """
        self.page.evaluate(f"() => {{\n{js_marker_script}\n}}")
