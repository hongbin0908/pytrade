
#!/usr/bin/env python
import os, sys, json, MySQLdb, datetime
from yahoo_finance import Share

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

if __name__ == '__main__':
    conn = MySQLdb.connect(host="123.56.128.198", user="root", passwd="123456", db="mytrade", charset="utf8")
    cursor = conn.cursor()

    backtrace = int(sys.argv[1])
    now = datetime.datetime.now()
    yahoo = Share('YHOO')
    for i in range(0,backtrace):
        d_cur_date = now - datetime.timedelta(days=i)
        s_cur_date = d_cur_date.strftime('%Y-%m-%d')
        print s_cur_date
        sql = "select * from exchange_days where datetime=%s"
        params = (s_cur_date)

        n = cursor.execute(sql, params)
        data =  yahoo.get_historical(s_cur_date, s_cur_date)
        if n == 0:
            sql = "insert into exchange_days(datetime, isvalid)" \
            +"values(%s,%s); "
            if len(data) > 0:
                params = (s_cur_date, 1)
            else:
                params = (s_cur_date, 0)
            n =cursor.execute(sql, params)
        else:
            sql = "update exchange_days set isvalid = %s where datetime=%s"
            if len(data) > 0:
                params = (1, s_cur_date)
            else:
                params = (0, s_cur_date)
            n=cursor.execute(sql, params)


    cursor.close() 
    conn.commit()
    conn.close()
