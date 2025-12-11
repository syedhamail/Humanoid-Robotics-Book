import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      collapsed: false,
      items: [
        'module1/chapter1',
        'module1/chapter2',
        'module1/chapter3',
        'module1/chapter4',
        'module1/chapter5',
        'module1/chapter6',
        'module1/chapter7',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Building Digital Twins',
      collapsed: false,
      items: [
        'module2/chapter1',
        'module2/chapter2',
        'module2/chapter3',
        'module2/chapter4',
        'module2/chapter5',
        'module2/chapter6',
        'module2/chapter7',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain with NVIDIA Isaac',
      collapsed: false,
      items: [
        'module3/chapter1',
        'module3/chapter2',
        'module3/chapter3',
        'module3/chapter4',
        'module3/chapter5',
        'module3/chapter6',
        'module3/chapter7',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Integration (Capstone)',
      collapsed: false,
      items: [
        'module4/chapter1',
        'module4/chapter2',
        'module4/chapter3',
        'module4/chapter4',
        'module4/chapter5',
        'module4/chapter6',
      ],
    },
  ],
};

export default sidebars;