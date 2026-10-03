import { type ReactNode, useEffect, useRef, useState } from 'react'
import { motion, type MotionValue, useScroll, useTransform } from 'framer-motion'
import '../containerScroll.css'

interface ContainerScrollProps {
  titleComponent: ReactNode
  children: ReactNode
}

// Ported from Aceternity UI's ContainerScroll (Next.js + Tailwind original) to
// plain CSS - this project has no Tailwind/shadcn, and pulling in Tailwind's
// base reset just for this component risked breaking the hand-styled chrome
// everywhere else. Motion logic is unchanged; only the className styling
// moved into containerScroll.css.
export function ContainerScroll({ titleComponent, children }: ContainerScrollProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({ target: containerRef })
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth <= 768)
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const scaleDimensions = (): [number, number] => (isMobile ? [0.7, 0.9] : [1.05, 1])

  const rotate = useTransform(scrollYProgress, [0, 1], [20, 0])
  const scale = useTransform(scrollYProgress, [0, 1], scaleDimensions())
  const translate = useTransform(scrollYProgress, [0, 1], [0, -100])

  return (
    <div className="cs-root" ref={containerRef}>
      <div className="cs-inner" style={{ perspective: '1000px' }}>
        <Header translate={translate} titleComponent={titleComponent} />
        <Card rotate={rotate} scale={scale}>
          {children}
        </Card>
      </div>
    </div>
  )
}

function Header({ translate, titleComponent }: { translate: MotionValue<number>; titleComponent: ReactNode }) {
  return (
    <motion.div style={{ translateY: translate }} className="cs-header">
      {titleComponent}
    </motion.div>
  )
}

function Card({
  rotate,
  scale,
  children,
}: {
  rotate: MotionValue<number>
  scale: MotionValue<number>
  children: ReactNode
}) {
  return (
    <motion.div
      style={{
        rotateX: rotate,
        scale,
        boxShadow:
          '0 0 #0000004d, 0 9px 20px #0000004a, 0 37px 37px #00000042, 0 84px 50px #00000026, 0 149px 60px #0000000a, 0 233px 65px #00000003',
      }}
      className="cs-card"
    >
      <div className="cs-card-inner">{children}</div>
    </motion.div>
  )
}
