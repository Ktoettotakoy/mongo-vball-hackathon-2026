import { AnimatePresence, motion } from 'framer-motion'
import { type ReactNode, useState } from 'react'

interface Tab {
  id: string
  title: string
  content: ReactNode
}

// Clickable "cards" that pick which panel shows below; switching plays a
// vertical slide, as if the clicked card pulls its content up into view.
export function CardTabs({ tabs }: { tabs: Tab[] }) {
  const [activeId, setActiveId] = useState(tabs[0]?.id)
  const active = tabs.find((t) => t.id === activeId) ?? tabs[0]

  return (
    <div className="card-tabs">
      <div className="card-tabs-row">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            className="card card-tab"
            data-active={t.id === activeId}
            onClick={() => setActiveId(t.id)}
          >
            <h2>{t.title}</h2>
          </button>
        ))}
      </div>

      <div className="card-tabs-panel-wrap">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={active?.id}
            initial={{ y: 28, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -28, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="card card-tabs-panel"
          >
            {active?.content}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
