import { html, TemplateResult } from 'lit';
import '../src/camera-shell.js';

export default {
  title: 'CameraShell',
  component: 'camera-shell',
  argTypes: {
    backgroundColor: { control: 'color' },
  },
};

interface Story<T> {
  (args: T): TemplateResult;
  args?: Partial<T>;
  argTypes?: Record<string, unknown>;
}

interface ArgTypes {
  header?: string;
  backgroundColor?: string;
}

const Template: Story<ArgTypes> = ({ header, backgroundColor = 'white' }: ArgTypes) => html`
  <camera-shell style="--camera-shell-background-color: ${backgroundColor}" .header=${header}></camera-shell>
`;

export const App = Template.bind({});
App.args = {
  header: 'My app',
};
