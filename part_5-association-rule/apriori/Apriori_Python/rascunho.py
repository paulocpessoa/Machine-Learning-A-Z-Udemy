lst = []
for i in range(0, len(results)):
    lst.append({'Support': results[i].support,
                'Base'= results[i].ordered_statistics[0].items_base})
