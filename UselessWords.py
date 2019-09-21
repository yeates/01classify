# -*- coding:utf-8 -*-
'''

Useless word profile, to delete residual information text

'''


operations = ['删除设置', '精华推荐', '取消置顶', '置顶', '关闭回复', '打开回复', '推荐编辑', '楼主禁言', '帖子标题', '精华理由']
illegal_symbol = ['&nbsp;']
all_regs = ['[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+',      # email address
            'http[s]?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',    # URL
            '(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s\d{1,2}:\d{1,2})', '(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # time stamp
            ]
