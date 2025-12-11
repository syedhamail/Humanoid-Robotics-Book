import React from 'react';
import type { ReactNode } from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Module 1: ROS 2 Fundamentals',
    Svg: require('@site/static/img/Module-1.svg').default,
    description: (
      <>
        "Module 1 introduces ROS 2 basics, covering nodes, topics, services, communication architecture, DDS,
        workspaces, and essential tools to build reliable, scalable robotics applications."
      </>
    ),
  },
  {
    title: 'Module 2: Building Digital Twins',
    Svg: require('@site/static/img/Module-2.svg').default,
    description: (
      <>
        "Module 2 teaches creating digital twins using simulation tools, modeling real-world systems, synchronizing
        virtual and physical data, enabling testing, monitoring, and optimizing robotic workflows efficiently."
      </>
    ),
  },
  {
    title: 'Module 3: AI-Robot Brain with NVIDIA Isaac',
    Svg: require('@site/static/img/Module-3.svg').default,
    description: (
      <>
        "Module 3 explores NVIDIA Isaac tools to build intelligent robot brains, integrating AI perception, navigation,
        simulation, and hardware acceleration for high-performance autonomous robotic systems."
      </>
    ),
  },
  {
    title: 'Module 4: Vision-Language-Action Integration (Capstone)',
    Svg: require('@site/static/img/Module-4.svg').default,
    description: (
      <>
        "Module 4 builds a complete capstone, integrating vision, language, and action models to create autonomous robots
        capable of understanding instructions, perceiving environments, and performing complex tasks."
      </>
    ),
  },
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center">
        <div className={styles.iconBox}>
          <Svg role="img" />
        </div>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3" style={{ marginTop: '0.5rem' }}>{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
